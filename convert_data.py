# convert_data.py
# This script reads multiple Kaggle CSV datasets, combines them, maps their labels,
# merges with our existing intents.json, and uses a powerful augmentation strategy
# to create our final, balanced, and robust dataset.

import pandas as pd
import json
import random

def augment_data(intents_dict, tag_to_augment, num_to_generate=1500):
    """
    Generates a large number of additional training patterns for a specific tag 
    using a wide variety of templates to ensure diversity.
    """
    print(f"Augmenting data for tag: '{tag_to_augment}'...")
    if tag_to_augment not in intents_dict:
        print(f"Warning: Tag '{tag_to_augment}' not found. Skipping.")
        return

    base_patterns = intents_dict[tag_to_augment]['patterns']
    
    # --- A much larger and more diverse set of templates ---
    templates = [
        "I'm feeling {}", "I feel so {}", "I've been dealing with a lot of {}",
        "Right now, I'm just {}", "My biggest issue is feeling {}", "I can't shake this feeling of being {}",
        "It's hard to explain, I'm just {}", "Lately, everything feels {}", "I think I'm a bit {}",
        "Can't seem to get over feeling {}", "It's like I'm stuck being {}", "All I feel is {}",
        "To be honest, I'm {}", "My mood is just {}", "Everything seems {}", "I'm in a {} state of mind",
        "It's just one of those days where I feel {}", "I'm experiencing a lot of {}"
    ]
    
    keywords = []
    for pattern in base_patterns:
        # A more robust way to get keywords by removing common starting phrases
        clean_pattern = pattern.lower()
        starters = ["i'm feeling", "i feel so", "i am", "i'm", "i feel"]
        for starter in starters:
            if clean_pattern.startswith(starter):
                clean_pattern = clean_pattern[len(starter):].strip()
        keywords.append(clean_pattern)
    
    if not keywords:
        print(f"No keywords found for tag '{tag_to_augment}'. Skipping.")
        return
        
    keywords = list(set(keywords))

    # Generate all possible unique combinations first
    all_possible_new_patterns = set()
    for template in templates:
        for keyword in keywords:
            all_possible_new_patterns.add(template.format(keyword))

    # Remove patterns that already exist in the base set
    all_possible_new_patterns.difference_update(base_patterns)

    # Decide how many to add
    if len(all_possible_new_patterns) < num_to_generate:
        patterns_to_add = list(all_possible_new_patterns)
    else:
        patterns_to_add = random.sample(list(all_possible_new_patterns), num_to_generate)

    # Add the new patterns to our dictionary
    intents_dict[tag_to_augment]['patterns'].extend(patterns_to_add)
    
    generated_count = len(patterns_to_add)
    print(f"Generated {generated_count} new patterns for '{tag_to_augment}'. Total patterns: {len(intents_dict[tag_to_augment]['patterns'])}")


def convert_kaggle_data():
    """
    Reads, processes, merges, and augments the dataset from multiple CSV files.
    """
    print("--- Starting Data Conversion Process ---")

    label_to_tag_map = {
        0: 'sadness', 1: 'gratitude', 2: 'gratitude',
        3: 'anger', 4: 'anxious'
    }
    print(f"Using label mapping: {label_to_tag_map}")

    print("Loading existing intents.json...")
    with open('intents.json', 'r', encoding='utf-8') as file:
        our_data = json.load(file)
    print("Existing data loaded successfully.")

    intents_dict = {intent['tag']: intent for intent in our_data['intents']}

    csv_files = ['training.csv', 'test.csv', 'validation.csv']
    all_dfs = []
    for csv_file in csv_files:
        try:
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {csv_file}.")
        except FileNotFoundError:
            print(f"\nWARNING: '{csv_file}' not found. Skipping this file.")
        except Exception as e:
            print(f"\nAn error occurred while reading {csv_file}: {e}")
    
    if not all_dfs:
        print("No Kaggle data files were loaded. Aborting.")
        return

    kaggle_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined Kaggle data contains {len(kaggle_df)} total rows.")

    print("Merging datasets...")
    new_patterns_count = 0
    for index, row in kaggle_df.iterrows():
        try:
            text = str(row['text'])
            label = int(row['label'])
            if label in label_to_tag_map:
                tag = label_to_tag_map[label]
                if tag in intents_dict:
                    intents_dict[tag]['patterns'].append(text)
                    new_patterns_count += 1
        except (KeyError, TypeError, ValueError):
            continue
    
    print(f"Successfully merged {new_patterns_count} new patterns from the Kaggle dataset.")

    # --- Augment data for ALL our under-represented intents ---
    print("\n--- Augmenting Data for Class Balance ---")
    augment_data(intents_dict, 'motivation')
    augment_data(intents_dict, 'confusion')
    augment_data(intents_dict, 'stressed')
    augment_data(intents_dict, 'neutral')
    augment_data(intents_dict, 'greeting')
    augment_data(intents_dict, 'goodbye')


    our_data['intents'] = list(intents_dict.values())

    with open('intents_expanded.json', 'w', encoding='utf-8') as file:
        json.dump(our_data, file, indent=2)

    print("\nNew, expanded, and fully balanced dataset saved to 'intents_expanded.json'.")
    print("--- Data Conversion Complete ---")


if __name__ == "__main__":
    convert_kaggle_data()
