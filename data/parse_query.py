import pandas as pd
import json
import os

def parse_query_csv(input_path="data/Query.csv", output_path="data/mab_kg.csv"):
    df = pd.read_csv(input_path)
    
    # Looking for:
    # mAb -> Target (isTargetOf or hasTarget or semanticscience.org/resource/SIO_000291)
    # mAb -> Indication (hasClinicalIndication)
    
    # Let's map URIs to simpler names, or just keep URIs.
    # For simplicity, we keep the last part of the URI.
    def get_basename(uri):
        if pd.isna(uri): return uri
        uri = str(uri)
        if "#" in uri:
            return uri.split("#")[-1]
        elif "/" in uri:
            return uri.split("/")[-1]
        return uri

    # Find hasClinicalIndication edges
    ind_edges = df[df['pred'].str.contains('hasClinicalIndication', na=False)]
    
    # We see in the preview:
    # "https://www.imgt.org/imgt-ontology#mAb_546","https://www.imgt.org/imgt-ontology#hasClinicalIndication","https://www.imgt.org/imgt-ontology#Cancers_gastric"
    # Actually wait, let's look at the preview:
    # "https://www.imgt.org/imgt-ontology#Cancers_gastric","https://www.imgt.org/imgt-ontology#isClinicalIndicationOf","https://www.imgt.org/imgt-ontology#mAb_546"
    
    mab_indications = []
    
    df_isInd = df[df['pred'].str.contains('isClinicalIndicationOf', na=False)]
    for _, row in df_isInd.iterrows():
        ind = get_basename(row['sub'])
        mab = get_basename(row['obj'])
        mab_indications.append({'mAb': mab, 'Indication': ind})
        
    df_hasInd = df[df['pred'].str.contains('hasClinicalIndication', na=False)]
    for _, row in df_hasInd.iterrows():
        mab = get_basename(row['sub'])
        ind = get_basename(row['obj'])
        mab_indications.append({'mAb': mab, 'Indication': ind})
        
    # Targets
    mab_targets = []
    df_isTarget = df[df['pred'].str.contains('isTargetOf', na=False)]
    for _, row in df_isTarget.iterrows():
        target = get_basename(row['sub'])
        mab = get_basename(row['obj'])
        mab_targets.append({'mAb': mab, 'Target': target})
        
    df_hasTarget = df[df['pred'].str.contains('hasTarget', na=False) | df['pred'].str.contains('SIO_000291', na=False)]
    for _, row in df_hasTarget.iterrows():
        mab = get_basename(row['sub'])
        target = get_basename(row['obj'])
        mab_targets.append({'mAb': mab, 'Target': target})
        
    # Merge them together into mab_kg.csv
    # We want rows of: mAb, Target, Indication
    df_ind = pd.DataFrame(mab_indications).drop_duplicates()
    df_tar = pd.DataFrame(mab_targets).drop_duplicates()
    
    if df_ind.empty and df_tar.empty:
        print("No edges found!")
        return
        
    if df_tar.empty:
        merged = df_ind
        merged['Target'] = None
    elif df_ind.empty:
        merged = df_tar
        merged['Indication'] = None
    else:
        merged = pd.merge(df_tar, df_ind, on='mAb', how='outer')
        
    # Exclude non-mAbs if any slipped in (like StudyProducts)
    merged = merged[merged['mAb'].str.startswith('mAb_') & ~merged['mAb'].str.startswith('StudyProduct')]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Extracted {len(merged)} edges and saved to {output_path}")
    print(merged.head())

if __name__ == "__main__":
    parse_query_csv()
