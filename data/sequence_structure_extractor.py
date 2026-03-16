import pandas as pd
import os
import requests
import json

class SequenceStructureExtractor:
    def __init__(self, data_path="data/mab_kg.csv"):
        self.data_path = data_path
        
    def load_kg_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found. Run kg_builder.py first.")
        return pd.read_csv(self.data_path)

    def fetch_sequence(self, mab_name):
        """
        Mock function to fetch sequence data. 
        In practice, this would query an IMGT API, TheraSAbDab, or Uniprot based on the mAb INN mapping.
        """
        # Dictionary of known mock sequences for testing
        mock_sequences = {
            "Pembrolizumab": {
                "heavy": "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
                "light": "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK"
            },
            "Trastuzumab": {
                "heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
                "light": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
            },
            "Rituximab": {
                "heavy": "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA",
                "light": "QIVLSQSPAILSASPGEKVTMTCRASSSVSYIHWFQQKPGSSPKPWIYATSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWTSNPPTFGGGTKLEIK"
            }
        }
        
        return mock_sequences.get(mab_name, {"heavy": "UNKNOWN", "light": "UNKNOWN"})

    def extract_features(self, output_path="data/mab_features.json"):
        df = self.load_kg_data()
        
        features = {}
        for idx, row in df.iterrows():
            mab_name = row['mAb']
            
            # Skip full URI names or parse them if necessary
            # For this mock, if the name starts with http, we use a generic name or dummy
            if mab_name.startswith("http"):
                mab_name = "Mock_mAb_" + str(idx)
                
            seqs = self.fetch_sequence(mab_name)
            features[mab_name] = {
                "target": row['Target'],
                "indication": row['Indication'],
                "heavy_chain": seqs['heavy'],
                "light_chain": seqs['light'],
                "structure_file": f"dummy_path/{mab_name}.pdb" # Placeholder for structure parsing
            }
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=4)
            
        print(f"Extracted sequences and saved to {output_path}")

if __name__ == "__main__":
    extractor = SequenceStructureExtractor()
    extractor.extract_features()
