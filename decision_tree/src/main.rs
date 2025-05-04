use decision_tree::{ read_csv, train, FeatureValue, predict };

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <csv_filename> <target_column>", args[0]);
        return;
    }

    let filename = &args[1];
    let target_column = &args[2];

    let (features, targets) = match read_csv(filename, target_column) {
        Ok((features, targets)) => (features, targets),
        Err(e) => {
            eprintln!("Failed to read CSV: {}", e);
            return;
        }
    };

    let feature_indices: Vec<usize> = (0..features.len()).collect();
    let result = train(features, targets, feature_indices);
    println!("Training successful! The decision tree has been created.");

    match result {
        Ok(tree) => {
            println!("***********************");
            println!("The tree is: {:?}", tree);
            println!("***********************");

            let test_instance = vec![
                FeatureValue::Categorical("Overcast".to_string()),
                FeatureValue::Categorical("Strong".to_string()),
                FeatureValue::Categorical("Mountain".to_string()),
                FeatureValue::Numeric(70),
                FeatureValue::Numeric(65),
            ];

            let predicted_label = predict(&tree, &test_instance);
            println!(
                "Predicted label for the instance {:?} is: {}",
                test_instance,
                predicted_label
            );
        }
        Err(e) => {
            eprintln!("Error during training: {}", e);
        }
    }
}
