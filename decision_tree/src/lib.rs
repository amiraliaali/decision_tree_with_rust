#![allow(non_snake_case)]

use csv::ReaderBuilder;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureValue {
    Numeric(i32),
    Categorical(String),
}

#[derive(Debug)]
pub enum Tree {
    Leaf(String),
    CategoricalNode {
        feature_index: usize,
        children: HashMap<FeatureValue, Tree>,
    },
    NumericNode {
        feature_index: usize,
        split_point: i32,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

/// Since the train() function was getting too large, we brought the categorical part in 
/// a separate function
pub fn train_categorical(targets: Vec<String>, features: Vec<Vec<FeatureValue>>, best_feature_index_in_features: usize, feature_indices: Vec<usize>) -> HashMap<FeatureValue, Tree>{
    // For each unique class, we need to have a child consisting of features and targets
    let mut subsets: HashMap<
        FeatureValue,
        (Vec<Vec<FeatureValue>>, Vec<String>)
    > = HashMap::new();

    for instance_idx in 0..targets.len() {
        let value_at_best_feature =
            features[best_feature_index_in_features][instance_idx].clone();

        // for each class of the categorical feature, create an entry in the hashmap and to it
        // initilaize two empty vectors, one for the features and one for the target
        let entry = subsets
            .entry(value_at_best_feature.clone())
            .or_insert_with(|| (vec![Vec::new(); features.len() - 1], Vec::new()));

        for (feature_idx, feature_column) in features.iter().enumerate() {
            if feature_idx == best_feature_index_in_features {
                continue;
            }
            let new_idx = if feature_idx < best_feature_index_in_features {
                feature_idx
            } else {
                feature_idx - 1
            };
            entry.0[new_idx].push(feature_column[instance_idx].clone());
        }

        entry.1.push(targets[instance_idx].clone());
    }

    let mut children = HashMap::new();
    for (feature_value, (sub_features, sub_targets)) in subsets {
        let mut new_feature_indices = feature_indices.clone();
        new_feature_indices.remove(best_feature_index_in_features);

        let child = train(sub_features, sub_targets, new_feature_indices);
        children.insert(feature_value, child.expect("Failed to build subtree"));
    }

    children
}


/// The function which fits the decision on the training data
pub fn train(
    features: Vec<Vec<FeatureValue>>,
    targets: Vec<String>,
    feature_indices: Vec<usize>
) -> Result<Tree, Box<dyn Error>> {
    // in case there are no more features left, return the most frequent target
    if features.is_empty() {
        if let Some(most_frequent_target) = get_most_frequent_target(&targets) {
            return Ok(Tree::Leaf(most_frequent_target));
        } else {
            return Err("Targets are empty and no majority target found.".into());
        }
    }

    assert_eq!(
        features[0].len(),
        targets.len(),
        "Mismatch between number of instances in features and targets."
    );

    // If targets are all the same, avoid splitting any further
    if check_if_targets_same(&targets) {
        return Ok(Tree::Leaf(targets[0].clone()));
    }

    let (best_feature_index_in_features, best_split_point) = best_discriminator_feature(
        features.clone(),
        targets.clone()
    );

    // We have a place holder for this, so that later when we split the features
    // we get the correct index in the original feature indexes
    let original_feature_index = feature_indices[best_feature_index_in_features];

    match &features[best_feature_index_in_features][0] {
        FeatureValue::Categorical(_) => {
            let children: HashMap<FeatureValue, Tree> = train_categorical(targets, features, best_feature_index_in_features, feature_indices);

            Ok(Tree::CategoricalNode {
                feature_index: original_feature_index,
                children,
            })
        }

        // since the nuemeric case is a bit more complicated, we didn't bring it in a separate
        // function
        FeatureValue::Numeric(_) => {
            let features_copy = features.clone();

            let mut left_features = vec![Vec::new(); features_copy.len()];
            let mut left_targets = Vec::new();
            let mut left_feature_indices: Vec<usize> = Vec::new();

            let mut right_features = vec![Vec::new(); features_copy.len()];
            let mut right_targets = Vec::new();
            let mut right_feature_indices: Vec<usize> = Vec::new();

            for instance_idx in 0..targets.len() {
                // extract the feature value
                let value: i32 = match &features[best_feature_index_in_features][instance_idx] {
                    FeatureValue::Numeric(v) => *v,
                    _ => {
                        return Err("Expected numeric feature.".into());
                    }
                };
                
                // if the current feature value is smaller than the spliting point, put the value alongside
                // its tagret to the left partition of the split else on right
                if value < best_split_point {
                    for (i, feature_column) in features_copy.iter().enumerate() {
                        left_features[i].push(feature_column[instance_idx].clone());
                        left_feature_indices.push(i);
                    }
                    left_targets.push(targets[instance_idx].clone());
                } else {
                    for (i, feature_column) in features_copy.iter().enumerate() {
                        right_features[i].push(feature_column[instance_idx].clone());
                        right_feature_indices.push(i);
                    }
                    right_targets.push(targets[instance_idx].clone());
                }
            }
            
            if left_targets.is_empty() || right_targets.is_empty(){
                if let Some(most_frequent_target) = get_most_frequent_target(&targets) {
                    return Ok(Tree::Leaf(most_frequent_target));
                } else {
                    return Err("Targets are empty and no majority target found.".into());
                }
            }

            let left_child = train(left_features, left_targets, left_feature_indices)?;
            let right_child = train(right_features, right_targets, right_feature_indices)?;

            Ok(Tree::NumericNode {
                feature_index: original_feature_index,
                split_point: best_split_point,
                left: Box::new(left_child),
                right: Box::new(right_child),
            })
        }
    }
}

/// A function that given a tree and an instance, containing a vector of feature values,
/// it predicts the output label based on the tree.
pub fn predict(tree: &Tree, instance: &Vec<FeatureValue>) -> String {
    match tree {
        Tree::Leaf(label) => return label.clone(),
        Tree::CategoricalNode { feature_index, children } => {
            let feature_value = &instance[*feature_index];
            match children.get(feature_value) {
                Some(subtree) => predict(subtree, instance),
                None => "Unknown".to_string(),
            }
        }
        Tree::NumericNode { feature_index, split_point, left, right } => {
            match &instance[*feature_index] {
                FeatureValue::Numeric(v) => {
                    if *v < *split_point {
                        predict(left, instance)
                    } else {
                        predict(right, instance)
                    }
                }
                _ => "Unknown".to_string(),
            }
        }
    }
}

/// From a vector of targets, it finds the most occuring label in them
fn get_most_frequent_target(targets: &Vec<String>) -> Option<String> {
    let mut frequency_map: HashMap<String, usize> = HashMap::new();

    // counts the occurance of each label and saves it in a hashmap
    for target in targets {
        *frequency_map.entry(target.clone()).or_insert(0) += 1;
    }

    let mut most_frequent_target: Option<String> = None;
    let mut count_most_frequent_target: usize = 0;

    for (key, value) in &frequency_map {
        if *value > count_most_frequent_target {
            count_most_frequent_target = *value;
            most_frequent_target = Some(key.clone());
        }
    }

    most_frequent_target
}

/// A function which finds the best feature to split on next.
/// It returns the index of the best feture to split on next.
pub fn best_discriminator_feature(
    features: Vec<Vec<FeatureValue>>,
    targets: Vec<String>
) -> (usize, i32) {
    let mut best_feature = 0;
    let mut best_information_gain: f64 = f64::NEG_INFINITY;
    let mut best_split_point: i32 = i32::MIN;

    let entropy_of_target = calculate_entropy(&targets);

    for (feature_index, feature) in features.iter().enumerate() {
        if feature.is_empty() {
            continue; // skip empty feature columns
        }
        let info_gain: f64;
        let overall_entropy: f64;
        let mut iteration_split_point: i32 = i32::MIN;

        // if the feature is categorical
        if matches!(feature[0], FeatureValue::Categorical(_)) {
            overall_entropy = overall_entropy_categorical_feature(feature, &targets);
        } else {
            // if the feature is numerical
            (overall_entropy, iteration_split_point) = overall_entropy_numerical_feature(
                feature,
                &targets
            );
        }

        info_gain = entropy_of_target - overall_entropy;

        if info_gain > best_information_gain {
            best_information_gain = info_gain;
            best_feature = feature_index;
            best_split_point = iteration_split_point;
        }
    }
    (best_feature, best_split_point)
}

/// For each numerical feature we calcualte the overall entropy.
/// We do this by first sorting the numerical feature vector alongside
/// the targets. In a for loop we each time go one step further and split
/// the dataset to two partitions and calcualte the entropy of the targets.
/// We return the split point which led to the smallest entropy
fn overall_entropy_numerical_feature(
    features: &Vec<FeatureValue>,
    targets: &Vec<String>
) -> (f64, i32) {
    let mut combined: Vec<(&FeatureValue, &String)> = features.iter().zip(targets.iter()).collect();

    combined.sort_by(|a, b| {
        match (a.0, b.0) {
            (FeatureValue::Numeric(val_a), FeatureValue::Numeric(val_b)) => {
                val_a.partial_cmp(val_b).unwrap()
            }
            _ => std::cmp::Ordering::Equal,
        }
    });

    let (sorted_features, sorted_targets): (Vec<&FeatureValue>, Vec<&String>) = combined
        .into_iter()
        .unzip();

    let mut min_entropy: f64 = f64::INFINITY;
    let mut best_split_value: i32 = i32::MIN;

    for (i, split_point) in sorted_features.iter().enumerate() {
        let split_value: i32 = match split_point {
            FeatureValue::Numeric(val) => *val,
            _ => panic!("Expected numeric"),
        };

        let left_targets: Vec<String> = sorted_targets[..i].iter().cloned().cloned().collect();
        let right_targets: Vec<String> = sorted_targets[i..].iter().cloned().cloned().collect();

        // We avoid to split in such a way that it gives an empty vector on one side
        if left_targets.is_empty() || right_targets.is_empty() {
            continue;
        }

        let prob_left: f64 = (left_targets.len() as f64) / (targets.len() as f64);
        let prob_right: f64 = (right_targets.len() as f64) / (targets.len() as f64);

        let entropy: f64 =
            prob_left * calculate_entropy(&left_targets) +
            prob_right * calculate_entropy(&right_targets);

        if entropy < min_entropy {
            min_entropy = entropy;
            best_split_value = split_value;
        }
    }
    (min_entropy, best_split_value)
}

/// For each feature we calcualte the overall entropy.
/// We do this by groupbying the targets based on different labels in each
/// feature, and then take their weighted sum.
/// prob_feture_1 * entropy_of_targets_splitted_on_feature_1 + ...
fn overall_entropy_categorical_feature(feature: &Vec<FeatureValue>, targets: &Vec<String>) -> f64 {
    let mut subsets: HashMap<String, Vec<String>> = HashMap::new();

    for (i, feature_value) in feature.iter().enumerate() {
        if let FeatureValue::Categorical(value) = feature_value {
            subsets.entry(value.clone()).or_insert(Vec::new()).push(targets[i].clone());
        }
    }

    let mut overall_entropy = 0.0;
    for subset in subsets.values() {
        let prob = (subset.len() as f64) / (targets.len() as f64);
        overall_entropy += prob * calculate_entropy(subset);
    }

    overall_entropy
}

/// Calculates the entorpy of a vector of strings.
/// We have had in different lectures that the entropy of a set is calculated
/// as the sum of : - prob_of_each_class * log2(prob_of_each_class)
fn calculate_entropy(targets: &Vec<String>) -> f64 {
    // a hash map to track the number of occurences of each target
    let mut counts: HashMap<String, i32> = HashMap::new();

    for target in targets {
        *counts.entry(target.clone()).or_insert(0) += 1;
    }

    let total: f64 = targets.len() as f64;
    let mut entropy: f64 = 0.0;

    for (_value, count) in counts {
        let prob: f64 = (count as f64) / total;
        entropy -= prob * prob.log2();
    }

    entropy
}

/// Checks whether or not all instances in a vector (in our case target vector)
/// are the same or not.
fn check_if_targets_same(targets: &Vec<String>) -> bool {
    if targets.is_empty() {
        return false;
    }

    let first_target: &String = &targets[0];

    for target in targets {
        if target != first_target {
            return false;
        }
    }

    true
}

/// Reads the csv and saves the records in two different vectors.
/// One for the fetures and one for the targets.
pub fn read_csv(
    path: &str,
    target_column: &str
) -> Result<(Vec<Vec<FeatureValue>>, Vec<String>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers = rdr.headers()?;

    let mut index_of_target_column: usize = 0;
    let mut found_target_column: bool = false;

    let mut index_of_feature_columns: Vec<usize> = Vec::new();

    // with this loop we find the index of the target- and the feature columns in the headers list
    for (i, header) in headers.iter().enumerate() {
        if header == target_column {
            index_of_target_column = i;
            found_target_column = true;
        } else {
            index_of_feature_columns.push(i);
        }
    }

    if found_target_column == false {
        return Err(
            format!("Target column '{}' not found in headers of the csv file.", target_column).into()
        );
    }

    // we separately save the values of the targets and features in two different vectors
    // the target should be categorical, the features could be numeric or categorical
    let mut targets: Vec<String> = Vec::new();
    let mut features: Vec<Vec<FeatureValue>> = vec![Vec::new(); index_of_feature_columns.len()];

    for result in rdr.records() {
        let record = result?;

        let target_value_string = record.get(index_of_target_column).unwrap_or("");
        let target_value = target_value_string.to_string();
        targets.push(target_value);

        for (index_feature, &col_idx) in index_of_feature_columns.iter().enumerate() {
            let feature_value_string = record.get(col_idx).unwrap_or("");
            features[index_feature].push(parse_value(feature_value_string));
        }
    }

    Ok((features, targets))
}

/// This function tries to parse the features that we have read sas string to integers (i32).
/// If it couldn't convert, it keeps them as string.
/// But we anyways parse each feature value in the struct that we have defined
fn parse_value(value: &str) -> FeatureValue {
    if let Ok(num) = value.parse::<i32>() {
        FeatureValue::Numeric(num)
    } else {
        FeatureValue::Categorical(value.to_string())
    }
}

