# Data Directory

This directory contains the datasets used for training and testing the lung cancer prognosis prediction models.

## File Structure

```
data/
├── Cli_CT_habitat1_PET_habitat1_train_OS.csv    # OS model training data (318 patients)
├── Cli_CT_habitat4_PET_habitat1_train_PFS.csv   # PFS model training data (318 patients)
├── sample_OS_data.csv                           # OS prediction sample data (11 patients)
├── sample_PFS_data.csv                          # PFS prediction sample data (11 patients)
└── README.md                                    # This file
```

## Dataset Description

### Training Data
- **318 patients** with complete clinical and imaging data
- **Multimodal features**: Clinical, CT radiomics, PET radiomics
- **Survival outcomes**: Overall Survival (OS) and Progression-Free Survival (PFS)

### Sample Data
- **11 patients** for demonstration and testing
- Same feature structure as training data
- Used for system validation and user tutorials

## Feature Categories

### Clinical Features
- **Age**: Patient age at diagnosis
- **Gender**: Patient gender (1=Male, 0=Female)
- **Cli**: Clinical staging information
- **Path**: Pathological staging information

### PET Parameters
- **TLG**: Total Lesion Glycolysis
- **MTV**: Metabolic Tumor Volume
- **SUVmax**: Maximum Standardized Uptake Value

### CT Radiomics Features (Habitat Analysis)
- **Original features**: Shape, first-order statistics
- **Wavelet features**: Multi-scale texture analysis
  - HLH, LLH, HHL patterns
- **Texture features**:
  - GLCM: Gray Level Co-occurrence Matrix
  - GLSZM: Gray Level Size Zone Matrix
  - NGTDM: Neighborhood Gray Tone Difference Matrix

### PET Radiomics Features (Habitat Analysis)
- **First-order features**: Statistical moments
- **Wavelet transforms**: Multi-resolution analysis
- **Log-sigma filters**: Scale-space analysis
- **Texture matrices**: GLCM, GLSZM, NGTDM features

## Data Preprocessing

All features have been:
1. **Standardized**: Z-score normalization applied
2. **Quality checked**: Missing values handled
3. **Feature selected**: Relevant features retained
4. **Validated**: Cross-validated for model training

## Model-Specific Features

### OS Model (25 features)
- Clinical: Age, Cli
- Metabolic: TLG, MTV, Path
- CT Habitat1: Shape and texture features
- PET Habitat1: Metabolic and texture features

### PFS Model (26 features)
- Clinical: Gender, Path
- Metabolic: TLG, MTV, SUVmax, Cli
- CT Habitat4: Advanced texture features
- PET Habitat1: Comprehensive radiomics features

## Usage Guidelines

### For Prediction
- Ensure your data matches the exact feature names
- All numeric features should be properly scaled
- Missing values should be handled before upload
- 'name' column should contain unique patient identifiers

### Data Format Example
```csv
name,Age,Cli,TLG,MTV,Path,[other features...]
Patient001,65.2,1.2,-0.45,0.33,-0.12,[values...]
Patient002,72.1,0.8,0.21,-0.15,0.67,[values...]
```

## Data Privacy

- All patient identifiers have been anonymized
- No personal health information is included
- Data used with appropriate ethical approval
- Training data contains de-identified research cohort

## Citation

If you use this dataset, please cite:
```
[Your publication citation here]
```

## Contact

For questions about the data or feature extraction methodology:
- Contact: [your.email@institution.edu]
- Documentation: See docs/ directory for detailed guides 