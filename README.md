# Forestal Monitoring & Carbon Sequestration

**Site Navigation:** 
- [Introduction](https://www.google.com/search?q=/index.html) 
- [Phase 1](https://www.google.com/search?q=/phase1.html) 
- [Phase 2](https://www.google.com/search?q=/phase2.html)  

**README Navigation:** 
- [Introduction](#introducton) 
- [Project Goals](#project-goals)
- [Methodology](#methodology)
- [Project Structure & Overview](#project-structure-&-overview)
---

## Introduction

The ongoing deforestation crisis in Paraguay is primarily driven by agricultural expansion and wildfires, leading to a rise in carbon emission levels, causing loss of biodiversity and an increase of the overall temperature in the region. Vital ecosystems are facing alarming rates of deforestation, the loss of which damages wildlife and reduces the region's capacity to capture and store carbon dioxide, further aggravating the global climate crisis.

Previous research made in deforestation monitoring has employed satellite images, geographic information and more to track changes in the forest over time. However, the use of **AI and machine learning** is a new set of opportunities for more accurate and dynamic tracking of the changes in the environment. This project aims to address the challenges of monitoring deforestation and carbon sequestration potential by developing an artificial intelligence model that is capable of following these issues in order to obtain important data about the damaged zones, enabling the identification of areas where reforestation efforts would yield the maximum carbon capture benefit.

The effects of deforestation are far-reaching. It leads to the loss of biodiversity, with species becoming extinct and ecosystems being disrupted. Additionally, deforestation contributes significantly to climate change by releasing large amounts of carbon stored in trees into the atmosphere.

## Project Goals

This initiative intends to provide:

* An accurate **deforestation detection system**.
* A **predictive model** for wildfire risk.
* A **map of carbon sequestration potential** across the region.

The project aims to contribute a valuable tool to make data-driven decisions for where to implement reforestation programs while raising awareness of deforestation and wildfire issues in Paraguay. The integration of this model is expected to significantly improve forest conservation efforts, leading to a substantial reduction in CO2 emissions.


## Methodology

This project makes use of AI and ML concepts to enhance deforestation detection methods and support reforestation in the Chaco region.


* **Data Integration:** Uses high-resolution satellite imagery from **[Sentinel 2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2)**.
* **Image Analysis:** Employs **Convolutional Neural Networks (CNNs)** for detailed image analysis.
* **Temporal Forecasting:** Applies **Recurrent Neural Networks (RNNs)** for monitoring vegetation changes and predicting trends.

The project focuses on identifying optimal locations for reforestation based on carbon sequestration potential, the final goal is to enable policymakers to prioritize the most impactful areas for intervention. The use of AI improves the accuracy of monitoring systems and provides actionable insights, offering a scalable, data-driven approach to fight deforestation.

This initiative aims to preserve Chaco’s ecosystems, reduce carbon emissions, and support informed decision-making in conservation, contributing to a sustainable balance between development and ecosystem preservation in fragile regions.

## Project Structure & Overview

[/images](./images/) and [/website](./website/) directories relate to the presentation of the project itself as a website. The [project report](./Forest_Monitoring_and_Carbon_Sequestration.pdf) is also included (in PDF format) as well asthe [dataset](./archive.zip) which can be found within a ZIP file.  

### Code related files  
- *[dataIni](./dataIni.py)* handles the extraction, loading, as well as a visualization of the dataset.  
- *[imageConversor](./imageConversor.py)* provides a Flask-based API for handling external image transformations.  
- *[processAndTrain](./processAndTrain.py)* calculates the [NDVI](https://www.earthdata.nasa.gov/topics/normalized-difference-vegetation-index-ndvi) to highlight biomass. Since standard imagery often lacks a Near-Infrared (NIR) band, the code approximates NDVI using the green channel:  
$$ \frac{Green - Red}{Green + Red + 1e-7} $$  
Images are loaded, normalized, and stacked into an *8-channel tensor* (2xRGB images + 2x NDVI images). The model architecture employed implements a *Sequential CNN with three convolutional layers and a softmax output for multiclass classification. The final weights are saved in the following format: *reforestation_model.h5*.  


---
