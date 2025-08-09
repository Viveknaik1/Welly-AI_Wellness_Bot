# convert_data.py
# This script processes the Kaggle CSV files and our custom intents.json.
# It merges them and uses data augmentation to balance the classes,
# creating a final, rich dataset for training the model.

import pandas as pd
import json
import random

def augment_data(intents_dict, tag_to_augment, num_to_generate=1500):
    """
    Generates additional training patterns for a specific tag to combat class imbalance.
    """
    print(f"Augmenting data for tag: '{tag_to_augment}'...")
    base_patterns = intents_dict.get(tag_to_augment, {}).get('patterns', [])
    if not base_patterns:
        print(f"Warning: No base patterns found for tag '{tag_to_augment}'. Skipping.")
        return

    augmented_patterns_set = set(base_patterns)

    templates = [
        "I'm feeling {}", "I feel so {}", "I've been dealing with a lot of {}",
        "Right now, I'm just {}", "My biggest issue is feeling {}", "I can't shake this feeling of being {}",
        "It's hard to explain, I'm just {}", "Lately, everything feels {}", "I think I'm a bit {}",
        "Can't seem to get over feeling {}", "It's like I'm stuck being {}", "All I feel is {}"
    ]
    
    keywords = list(set(p.lower().replace("i'm feeling", "").replace("i feel", "").strip() for p in base_patterns))

    if not keywords:
        print(f"No keywords found for tag '{tag_to_augment}'. Skipping.")
        return

    # Generate all possible unique combinations to avoid getting stuck in a loop
    all_possible_new_patterns = {template.format(keyword) for template in templates for keyword in keywords}
    all_possible_new_patterns.difference_update(base_patterns)

    # Add a sample of the new patterns to the dataset
    patterns_to_add = random.sample(list(all_possible_new_patterns), min(len(all_possible_new_patterns), num_to_generate))
    intents_dict[tag_to_augment]['patterns'].extend(patterns_to_add)
    
    print(f"Generated {len(patterns_to_add)} new patterns for '{tag_to_augment}'.")

def convert_kaggle_data():
    """
    Main function to read, process, merge, and augment the datasets.
    """
    print("--- Starting Data Conversion Process ---")

    label_to_tag_map = {0: 'sadness', 1: 'gratitude', 2: 'gratitude', 3: 'anger', 4: 'anxious'}

    with open('intents.json', 'r', encoding='utf-8') as file:
        our_data = json.load(file)
    intents_dict = {intent['tag']: intent for intent in our_data['intents']}

    # Load and combine all Kaggle CSV files
    csv_files = ['training.csv', 'test.csv', 'validation.csv']
    all_dfs = [pd.read_csv(f) for f in csv_files if pd.io.common.file_exists(f)]
    
    if not all_dfs:
        print("No Kaggle data files found. Aborting.")
        return

    kaggle_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined Kaggle data contains {len(kaggle_df)} total rows.")

    # Merge the Kaggle data into our intents dictionary
    for _, row in kaggle_df.iterrows():
        try:
            tag = label_to_tag_map.get(int(row['label']))
            if tag and tag in intents_dict:
                intents_dict[tag]['patterns'].append(str(row['text']))
        except (KeyError, TypeError, ValueError):
            continue
    
    # Augment data for our custom, under-represented intents
    print("\n--- Augmenting Data for Class Balance ---")
    custom_intents = ['motivation', 'confusion', 'stressed', 'neutral', 'greeting', 'goodbye']
    for tag in custom_intents:
        augment_data(intents_dict, tag)

    # Save the final, combined dataset
    our_data['intents'] = list(intents_dict.values())
    with open('intents_expanded.json', 'w', encoding='utf-8') as file:
        json.dump(our_data, file, indent=2)

    print("\nNew, expanded, and balanced dataset saved to 'intents_expanded.json'.")
    print("--- Data Conversion Complete ---")

if __name__ == "__main__":
    convert_kaggle_data()
