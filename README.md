# imputation-AR6-carbon-sequestration

This repository is linked to the following preprint:

Prütz, R., Fuss, S., and Rogelj, J.: Imputation of missing IPCC AR6 data on land carbon sequestration, Earth Syst. Sci. Data Discuss. [preprint], https://doi.org/10.5194/essd-2024-68, in review, 2024.

This repository includes: 
- An imputation dataset for missing land carbon sequestation data of the AR6 Scenario Database for global scenarios and R10 scenario variants
- Code to test, compare and visualize the performance of regression models to predict missing land removal data
- Code to compare and visualize available AR6 land removal data and existing AR6 data reanalyses

The following two datasets are required to replicate the analysis:
- Byers, E., Krey, V., Kriegler, E., Riahi, K., Schaeffer, R., Kikstra, J., Lamboll, R., Nicholls, Z., Sandstad, M., Smith, C., van der Wijst, K., Al -Khourdajie, A., Lecocq, F., Portugal-Pereira, J., Saheb, Y., Stromman, A., Winkler, H., Auer, C., Brutschin, E., … van Vuuren, D. (2022). AR6 Scenarios Database [Data set]. In Climate Change 2022: Mitigation of Climate Change (1.1). Intergovernmental Panel on Climate Change. https://doi.org/10.5281/zenodo.7197970
- Gidden, M., Gasser, T., Grassi, G., Forsell, N., Janssens, I., Lamb, W. F., Minx, J., Nicholls, Z., Steinhauser, J., & Riahi, K. (2023). Dataset for Gidden et.al. 2023 Updated AR6 Mitigation Benchmarks using National Emissions Inventories (Version v2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10158920
The variable imputation is based on the dataset by Byers et al. (2022). The dataset by Gidden et al. (2023) is used for variable comparison. 
