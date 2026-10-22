import pandas as pd



input_csv_path = 'penguins.csv'          
output_csv_path = 'Ready_penguins.csv' 

def prepare_penguin_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
   
    for col in ['Species', 'OriginLocation']:
        if col in df.columns:
            mode_result = df[col].mode()
            fill_value=mode_result[0] if not mode_result.empty else 'Unknown'
            df[col] = df[col].fillna(fill_value)

    
    def get_first_mode(series):
        return series.mode().iloc[0] if not series.mode().empty else pd.NA
    
    # Amla el cells elfadia
    numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    
    for col in numerical_cols:
        if col in df.columns:
            # Emla baktar haga zaharat
            species_modes = df.groupby('Species')[col].transform(get_first_mode)
            df[col] = df[col].fillna(species_modes)
        
            if df[col].isna().any():
                global_fallback = df[col].mode().iloc[0] if not df[col].mode().empty else df[col].median()
                df[col] = df[col].fillna(global_fallback)

    # 4. one hot encoded 'OriginLocation'
    if 'OriginLocation' in df.columns:
        df = pd.get_dummies(df, columns=['OriginLocation'], drop_first=False)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

    # khally el cells men zero ela one alashan elmodel mayakodsh akbar column ka kyam hoo elassasy
    scale_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    for col in scale_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                df[col] = 0.0
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    
    if 'Species' in df.columns:
        df['Species'] = df['Species'].astype('category').cat.codes

   
    df.to_csv(output_path, index=False)
   

if __name__ == '__main__':
    prepare_penguin_data(input_path=input_csv_path, output_path=output_csv_path)