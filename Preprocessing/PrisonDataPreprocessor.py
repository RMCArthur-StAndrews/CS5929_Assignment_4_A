"""
Prison Data Preprocessing Module

This module provides a class for preprocessing prison and prisoner data
for constraint satisfaction modeling tasks.
"""

import json
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, List, Tuple, Optional


class PrisonDataPreprocessor:
    """
    A class to handle preprocessing of prison and prisoner data.
    
    @property encoder - OrdinalEncoder: Sklearn encoder for categorical variables
    @property col_mapping - Dict: Mapping of categorical columns to encoded values
    @property non_numerical_cols - pd.Index: Index of non-numerical column names
    """
    
    def __init__(self):
        """Initialize the PrisonDataPreprocessor."""
        self.encoder = OrdinalEncoder()
        self.col_mapping = {}
        self.non_numerical_cols = None
        
    def load_prison_template(self, filepath: str) -> pd.DataFrame:
        """
        Load and flatten prison template data from JSON file.
        
        @param filepath - str: Path to the prison template JSON file
        @returns pd.DataFrame: Flattened prison data with bed records
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        prison = data['prison']
        bed_records = []
        
        for section in prison['sections']:
            section_id = section['section_id']
            section_name = section['section_name']
            age_category = section['age_category']
            
            for ward in section['wards']:
                ward_id = ward['ward_id']
                supervision_level = ward['supervision_level']
                
                for wing in ward['wings']:
                    wing_id = wing['wing_id']
                    sex_assignment = wing['sex_assignment']
                    
                    for cell in wing['cells']:
                        cell_id = cell['cell_id']
                        cell_type = cell['cell_type']
                        
                        for bed in cell['beds']:
                            bed_record = {
                                'section_id': section_id,
                                'section_name': section_name,
                                'age_category': age_category,
                                'ward_id': ward_id,
                                'supervision_level': supervision_level,
                                'wing_id': wing_id,
                                'sex_assignment': sex_assignment,
                                'cell_id': cell_id,
                                'cell_type': cell_type,
                                'bed_id': bed['bed_id'],
                                'occupied': bed['occupied'],
                                'prisoner_id': bed['prisoner_id']
                            }
                            bed_records.append(bed_record)
        
        return pd.DataFrame(bed_records)
    
    def load_prisoner_list(self, filepath: str, key: str = "incoming_prisoners", 
                          limit: Optional[int] = None) -> List[Dict]:
        """
        Load prisoner list from JSON file.
        
        @param filepath - str: Path to the prisoner list JSON file
        @param key - str: Key in JSON to access prisoner list
        @param limit - int, optional: Maximum number of prisoners to load
        @returns List[Dict]: List of prisoner dictionaries
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        prisoners = data[key]
        
        if limit is not None:
            prisoners = prisoners[:limit]
            
        return prisoners
    
    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode non-numerical columns in a DataFrame.
        
        @param df - pd.DataFrame: DataFrame to encode
        @returns pd.DataFrame: Encoded DataFrame
        """
        df_encoded = df.copy()
        self.non_numerical_cols = df.select_dtypes(include=['object', 'bool']).columns
        df_encoded[self.non_numerical_cols] = self.encoder.fit_transform(df[self.non_numerical_cols])
        
        # Create mapping for later use
        self._create_column_mapping()
        
        return df_encoded
    
    def _create_column_mapping(self):
        """
        Create mapping from categorical values to encoded integers.
        """
        self.col_mapping = {}
        for i, col in enumerate(self.non_numerical_cols):
            categories = self.encoder.categories_[i]
            self.col_mapping[col] = {cat: idx for idx, cat in enumerate(categories)}
    
    def encode_prisoners(self, prisoners: List[Dict]) -> List[Dict]:
        """
        Encode prisoner data using the existing encoder mapping.
        
        @param prisoners - List[Dict]: List of prisoner dictionaries
        @returns List[Dict]: List of encoded prisoner dictionaries
        @raises ValueError: If encoder has not been fitted yet
        """
        if not self.col_mapping:
            raise ValueError("Encoder must be fitted first. Call encode_dataframe() before encoding prisoners.")
        
        # Determine which sex column name exists in the mapping
        sex_col = 'sex_assignment' if 'sex_assignment' in self.col_mapping else 'sex'
        
        prisoners_encoded = []
        for prisoner in prisoners:
            prisoner_encoded = prisoner.copy()
            prisoner_encoded['prisoner_id'] = int(prisoner['prisoner_id'].lstrip('P'))
            
            # Map categorical values using the same encoding
            prisoner_encoded['age_category_encoded'] = self.col_mapping['age_category'][prisoner['age_category']]
            prisoner_encoded['sex_encoded'] = self.col_mapping[sex_col][prisoner['sex']]
            prisoner_encoded['supervision_level_encoded'] = self.col_mapping['supervision_level'][prisoner['supervision_level']]
            prisoners_encoded.append(prisoner_encoded)
        
        return prisoners_encoded
    
    def merge_prisoners_with_beds(self, prisoners_df: pd.DataFrame, 
                                 beds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge prisoner and bed DataFrames on prisoner_id.
        
        @param prisoners_df - pd.DataFrame: DataFrame of prisoners
        @param beds_df - pd.DataFrame: DataFrame of beds
        @returns DataFrame: Merged and cleaned DataFrame
        """
        merged_df = pd.merge(prisoners_df, beds_df, 
                            left_on='prisoner_id', 
                            right_on='prisoner_id', 
                            how='inner')
        
        # Deduplicate on bed_id (data quality guardrail: a bed should have exactly one occupant)
        duplicate_beds = merged_df[merged_df['bed_id'].duplicated(keep=False)]
        if not duplicate_beds.empty:
            # Keep the first occurrence per bed_id to avoid conflicting occupants
            merged_df = merged_df.drop_duplicates(subset=['bed_id'], keep='first')
        
        # Drop duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Drop unnecessary columns (but preserve boolean columns)
        columns_to_drop = ['sex_assignment', 'age_category_x', 
                          'supervision_level_x', 'cell_id', 'occupied']
        merged_df = merged_df.drop(columns=[col for col in columns_to_drop 
                                           if col in merged_df.columns])
        
        # Rename columns
        merged_df = merged_df.rename(columns={
            'age_category_y': 'age_category',
            'supervision_level_y': 'supervision_level'
        })
        
        return merged_df
    
    def prepare_conjure_mappings(self, prisoners_encoded: List[Dict], 
                                df_encoded: pd.DataFrame,
                                use_time_served: bool = False,
                                df_original: pd.DataFrame = None) -> Dict:
        """
        Prepare data mappings for Conjure constraint solver.
        
        @param prisoners_encoded - List[Dict]: List of encoded prisoner dictionaries
        @param df_encoded - pd.DataFrame: Encoded DataFrame with bed/prisoner data
        @param use_time_served - bool: Whether to include time_served data
        @param df_original DataFrame: Original unencoded DataFrame (required if use_time_served=True)
        @returns results: Dictionary containing all mappings and ranges for Conjure
        """
        # Prisoner mappings
        incoming_prisoners_sex = {}
        incoming_prisoners_age = {}
        incoming_prisoners_supervision = {}
        prisoner_ids = []
        
        for prisoner in prisoners_encoded:
            prisoner_id = int(prisoner['prisoner_id'])
            prisoner_ids.append(prisoner_id)
            incoming_prisoners_sex[prisoner_id] = int(prisoner['sex_encoded'])
            incoming_prisoners_age[prisoner_id] = int(prisoner['age_category_encoded'])
            incoming_prisoners_supervision[prisoner_id] = int(prisoner['supervision_level_encoded'])
        
        # Bed mappings
        bed_data = self._extract_bed_mappings(df_encoded, use_time_served, df_original)
        
        # Create result dictionary
        result = {
            'incoming_prisoners_sex': incoming_prisoners_sex,
            'incoming_prisoners_age': incoming_prisoners_age,
            'incoming_prisoners_supervision': incoming_prisoners_supervision,
            'prisoner_min': min(prisoner_ids),
            'prisoner_max': max(prisoner_ids),
            **bed_data
        }
        
        return result
    
    def _extract_bed_mappings(self, df_encoded: pd.DataFrame, 
                             use_time_served: bool = False,
                             df_original: pd.DataFrame = None) -> Dict:
        """
        Extract bed mappings from encoded DataFrame.
        
        @param df_encoded - pd.DataFrame: Encoded DataFrame
        @param use_time_served - bool: Whether to include time_served data
        @param df_original - pd.DataFrame: Original unencoded DataFrame (required if use_time_served=True)
        @returns result: Dictionary of bed mappings
        """
        bed_sex = {}
        bed_age = {}
        life_without_parole = {}
        untried = {}
        terrorism = {}
        subject_to_removal = {}
        supervised_release = {}
        sexual_or_domestic_harm = {}
        non_harassment_order = {}
        bed_supervision = {}
        bed_time_served = {} if use_time_served else None
        
        # Determine which column name to use for sex (check once, not per row)
        sex_col = 'sex_assignment' if 'sex_assignment' in df_encoded.columns else 'sex'
        
        # Build a mapping from encoded bed_id to original time_served using matching indices
        bed_id_to_time_served = {}
        if use_time_served and df_original is not None:
            # Both dataframes should have the same indices and same row order
            for idx in df_encoded.index:
                encoded_bed_id = int(df_encoded.loc[idx, 'bed_id'])
                original_time_served = int(df_original.loc[idx, 'time_served'])
                bed_id_to_time_served[encoded_bed_id] = original_time_served
        
        for index, row in df_encoded.iterrows():
            bed_id = int(row['bed_id'])
            
            bed_sex[bed_id] = int(row[sex_col])
            bed_age[bed_id] = int(row['age_category'])
            bed_supervision[bed_id] = int(row['supervision_level'])
            life_without_parole[bed_id] = int(row.get('is_life_without_parole', 0))
            untried[bed_id] = int(row.get('is_untried', 0))
            terrorism[bed_id] = int(row.get('is_terrorism', 0))
            subject_to_removal[bed_id] = int(row.get('is_subject_to_removal', 0))
            supervised_release[bed_id] = int(row.get('is_supervised_release', 0))
            sexual_or_domestic_harm[bed_id] = int(row.get('is_sexual_or_domestic_harm', 0))
            non_harassment_order[bed_id] = int(row.get('is_non_harassment_order', 0))
            
            # Use the pre-built mapping
            if use_time_served:
                if bed_id in bed_id_to_time_served:
                    bed_time_served[bed_id] = bed_id_to_time_served[bed_id]
                else:
                    raise ValueError(f"time_served data not found for bed_id {bed_id}")
        
        result = {
            'bed_sex': bed_sex,
            'bed_age': bed_age,
            'bed_supervision': bed_supervision,
            'bed_min': int(df_encoded['bed_id'].min()),
            'bed_max': int(df_encoded['bed_id'].max()), 
            'is_life_without_parole': life_without_parole,
            'is_untried': untried,
            'is_terrorism': terrorism,
            'is_subject_to_removal': subject_to_removal,
            'is_supervised_release': supervised_release,
            'is_sexual_or_domestic_harm': sexual_or_domestic_harm,
            'is_non_harassment_order': non_harassment_order
        }
        
        if bed_time_served is not None:
            result['bed_time_served'] = bed_time_served
        
        return result
    
    def prepare_allocation_mappings(self, prisoners_encoded: List[Dict], 
                                   empty_beds_df: pd.DataFrame) -> Dict:
        """
        Prepare mappings for prison allocation task with sequential bed IDs.
        
        @param prisoners_encoded - List[Dict]: List of encoded prisoner dictionaries
        @param empty_beds_df - pd.DataFrame: DataFrame containing only empty beds
        @returns results: Dictionary containing all mappings and ranges for allocation
        """
        # Prisoner mappings
        incoming_prisoners_sex = {}
        incoming_prisoners_age = {}
        incoming_prisoners_supervision = {}
        prisoner_ids = []
        
        for prisoner in prisoners_encoded:
            prisoner_id = int(prisoner['prisoner_id'])
            prisoner_ids.append(prisoner_id)
            incoming_prisoners_sex[prisoner_id] = int(prisoner['sex_encoded'])
            incoming_prisoners_age[prisoner_id] = int(prisoner['age_category_encoded'])
            incoming_prisoners_supervision[prisoner_id] = int(prisoner['supervision_level_encoded'])
        
        # Bed mappings with sequential IDs
        prison_beds_sex = {}
        prison_beds_age = {}
        prison_beds_supervision = {}
        bed_counter = 1
        
        for index, row in empty_beds_df.iterrows():
            prison_beds_sex[bed_counter] = int(row['sex_assignment'])
            prison_beds_age[bed_counter] = int(row['age_category'])
            prison_beds_supervision[bed_counter] = int(row['supervision_level'])
            bed_counter += 1
        
        return {
            'incoming_prisoners_sex': incoming_prisoners_sex,
            'incoming_prisoners_age': incoming_prisoners_age,
            'incoming_prisoners_supervision': incoming_prisoners_supervision,
            'prison_beds_sex': prison_beds_sex,
            'prison_beds_age': prison_beds_age,
            'prison_beds_supervision': prison_beds_supervision,
            'prisoner_min': min(prisoner_ids),
            'prisoner_max': max(prisoner_ids),
            'bed_min': 1,
            'bed_max': bed_counter - 1
        }
    
    def decode_bed_attributes(self, bed_row: pd.Series) -> Dict[str, str]:
        """
        Decode encoded bed attributes back to original categorical values.
        
        @param bed_row - pd.Series: Row from encoded DataFrame
        @returns bed attributes: Dictionary with decoded attributes
        """
        non_numerical_list = list(self.non_numerical_cols)
        
        bed_sex = self.encoder.categories_[non_numerical_list.index('sex_assignment')][int(bed_row['sex_assignment'])]
        bed_age = self.encoder.categories_[non_numerical_list.index('age_category')][int(bed_row['age_category'])]
        bed_supervision = self.encoder.categories_[non_numerical_list.index('supervision_level')][int(bed_row['supervision_level'])]
        
        return {
            'sex': bed_sex,
            'age_category': bed_age,
            'supervision_level': bed_supervision
        }
