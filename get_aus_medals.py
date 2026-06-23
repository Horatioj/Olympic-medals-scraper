import pandas as pd
import verify_medal_counts_from_positions as vmc
import aggregate_olympedia_results as aor

def main():
    df = pd.read_csv('data/olympedia_event_level_2000_2024.csv', dtype=str)
    
    # Filter for 2000 to 2022
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[(df['year_num'] >= 2000) & (df['year_num'] <= 2022)]
    
    # Filter for Australia (AUS)
    df = df[df['participating_noc'] == 'AUS'].copy()
    
    # Exclude Mixed and Open/Unknown events
    # df = df[df['event_gender'].isin(['Men', 'Women'])].copy()
    
    # Apply position logic
    placements = df['position'].map(vmc.placements_from_position)
    df['gold'] = placements.map(lambda vals: vmc.tally_medals(vals)[0])
    df['silver'] = placements.map(lambda vals: vmc.tally_medals(vals)[1])
    df['bronze'] = placements.map(lambda vals: vmc.tally_medals(vals)[2])
    
    # Calculate gender weights
    df['female_proportion'] = pd.to_numeric(df['female_proportion'], errors='coerce')
    weights = df.apply(lambda r: aor.gender_score_weights(r['event_gender'], r['female_proportion']), axis=1)
    df['female_weight'] = [w[0] for w in weights]
    df['male_weight'] = [w[1] for w in weights]
    
    # Distribute medals
    df['gold_female'] = df['gold'] * df['female_weight']
    df['silver_female'] = df['silver'] * df['female_weight']
    df['bronze_female'] = df['bronze'] * df['female_weight']
    df['total_female'] = df['gold_female'] + df['silver_female'] + df['bronze_female']
    
    df['gold_male'] = df['gold'] * df['male_weight']
    df['silver_male'] = df['silver'] * df['male_weight']
    df['bronze_male'] = df['bronze'] * df['male_weight']
    df['total_male'] = df['gold_male'] + df['silver_male'] + df['bronze_male']
    
    print("Total Medals by Gender for Australia (2000-2022), correctly allocating Mixed/Open events:")
    print(f"Women - Gold: {df['gold_female'].sum():g}, Silver: {df['silver_female'].sum():g}, Bronze: {df['bronze_female'].sum():g}, Total: {df['total_female'].sum():g}")
    print(f"Men   - Gold: {df['gold_male'].sum():g}, Silver: {df['silver_male'].sum():g}, Bronze: {df['bronze_male'].sum():g}, Total: {df['total_male'].sum():g}")

if __name__ == '__main__':
    main()
