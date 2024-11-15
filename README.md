# marine-microorganisms-clustering

Abstract: The objective in this analysis is to group microorganisms according to their type by utilizing state-ofthe-art clustering algorithms. This will allow us to better understand the different species in the dataset and the various characteristics that define them. The dataset comprises more than 1 million unannotated images gathered from the ocean by marine biologists associated with the Weizmann Institute. We will also closely examine the data’s temporal trends to identify patterns that may emerge over time. In addition to our primary objectives, we will perform various secondary analyses to explore the dataset further. This may include visualizing the data in different ways to gain new perspectives or utilizing machine learning techniques to identify complex patterns that may be difficult to discern through traditional analysis methods.

Data Pre-Processing
Data analysis and modeling processes require accurate and high-quality input data. In the case of image analysis, it is essential to ensure that the images being used are of adequate size and resolution for the task at hand. The raw data for this particular project consists of a single image comprising several cells. In order to prepare this data for analysis, our first step is to split each image into individual images, each containing only a single organism. This will allow us to isolate and analyze each cell separately. Moreover, the images are relatively small with various sizes, for example, some images contain only 20Π20 pixels. This is inadequate for a pre-trained vision model to analyze the cells in the image accurately. Therefore, we changed the size of each image to be 64Π64 pixels using standard image transformations provided by Pytorch [7]. This will ensure that the cells are large enough for the model to detect features and patterns accurately.
Extract Meaningful Representations
By leveraging a pre-trained DINO model [1] trained on ImageNet [3], we derived rich and meaningful image representations. Since the distribution of the images that were used to train this model is significantly different than our data, we inspect the attention maps (Figure 3) obtained by the pre-trained model on our data. To enhance the model’s performance we additionally fine-tuned this model on our specific dataset, for one epoch. Thus we extracted the most informative features relevant to our images.

Data
The dataset utilized in this study was obtained through an experiment that was specifically designed to investigate the growth dynamics of protists, which are eukaryotic single-cell organisms found in the fjord water of Bergen, Norway. To conduct the experiment, seven bags were submerged in the fjord water and water sampleswere collected on a daily basis for further analysis.One of the main objectives of the study was to gaininsight into the diversity of protists present throughoutthe experiment. To achieve this goal, 10mL of seawaterwas analyzed using the FlowCAM II™ instrument,which generates an image composed of manysingle particle images (Figure 1) and a metafile containingthe coordinates for each image in the raw data.A script was used to extract the single cell imagesfrom the larger composite image (Figure 2). As previouslymentioned, the individual particle images arerelatively small, with an average size of 37Π37 pixels. Ten thousand single-particle images were collectedfrom the seven bags and the ocean for 23 days,resulting in a dataset containing over 1.2 million images.In order to train our model, we will utilize theimages that have been acquired. This approach allowedus to effectively train and test our model on adiverse data set, providing us with a more comprehensive understanding of the protist populations within thefjord water.The raw cell images are very small compared tostandard benchmarks and contain only limited information. Since current off-the-shelf pre-trained modelsare fitted to larger images, it is difficult to extract interestinginsights.

![image](https://github.com/user-attachments/assets/4a2ae41d-1f45-4f45-a4e1-cb7d54237d99)

Figure 1: An output image of the FlowCAM II instrument

![image](https://github.com/user-attachments/assets/f3271370-1c91-4621-b1eb-6dcfb458a396)

Figure 2: Single particle image after splitting

Experiments and Analysis
In order to evaluate the effectiveness of our analysis,we conducted a thorough evaluation of each step weundertook. To begin, we pre-processed the images, with the goal of obtaining single images such as shown in figure 2. This process involved resizing all images to 64x64 pixels, as we chose to work with the DINO model, and use an 8x8 patch size. After preprocessing, we evaluated the pre-trained DINO modelon this data. We evaluated it in various ways including generating a visualization of the attention mapslearned by the model and visualizing the embedded space after applying UMAP reduction. This allowed us to gain a better understanding of the crucial features the model considers to produce its output, and its ability to distinguish different organisms. Upon analyzing these results produced by the pre-trained model, we realized that these results are not sufficient and that further improvement is needed. Hence, we fine-tunedthe model on our dataset for a single epoch and reanalyzed the fine-tuned model. The results were much better, and the finetuned model was more suitable for our dataset compared to the pre-trained model (see figure 3).
In addition, we also evaluated the separability of the embedding space produced by the two models. While the pre-trained model was able to distinguish only a few clusters, the fine-tuned model exhibits more distinguished clusters (figure 4).

![image](https://github.com/user-attachments/assets/c6270015-42bf-463a-b7d7-e01047e98f7f)

Figure 3: Comparison of the pre-trained and the fine-tuned models based on A) Original image and the attention maps of the B) pre-trained and C) the fine-tuned model

more examples:

![image](https://github.com/user-attachments/assets/087786e2-de15-4a95-aa80-2badf301067b)

![image](https://github.com/user-attachments/assets/6b49debb-d68f-4418-ad7a-3ad8ad9eba5b)

Figure 4: HDBSCAN result applied on the UMAP reduction based on the fine-tuned model’s representations

The clusters were produced via applying the HDBSCAN algorithm. After performing the clustering analysis on our dataset, we proceeded to investigate the nature of the clusters we had obtained. Our objective was to determine if the clusters comprised similar images or if they represented the same organisms. Through our analysis, we discovered that certain clusters were indeed composed of images that shared common characteristics, while other clusters actually represented organisms that had the same shape (figure 5) The results of our clustering analysis yielded intriguing insights into the underlying patterns present within our data. 

![image](https://github.com/user-attachments/assets/d7db7ee2-066a-4bd5-84ed-6b007c1111bb)

Figure 5: Random images samples from the different clusters. This figure shows 4 clusters detected by our method, and for each cluster, 16 images were randomly sampled. As can be seen, each cluster consists of different visual characteristics. Our method was able to distinguish different organisms by their visual components.

Specifically, we observed that some clusters were more likely to appear in the initial days of the experiment, while others were representative of the later days (figure 6). This dynamic clustering behavior could potentially provide valuable information on how different organisms succeed one another in an indeterminate manner. By uncovering these distinct patterns, we can gain a deeper understanding of the biological processes at play within our dataset. Moreover, we had previous knowledge about when each image was sampled and from which bag, enablingus to assess if the models’ representations clustered differently by the day or the bag. Upon comparingthe pre-trained and fine-tuned models, we discoveredthat the fine-tuned model performed better, indicatingthe success of the learning process.

![image](https://github.com/user-attachments/assets/dfa32de3-6c51-4faa-84bb-d543d768dc33)

Figure 6: Cluster distribution by day of the experiment. Each column corresponds to a different cluster, each row to a different day. The figure shows the distribution of each cluster on different days. Each column is summed to 1. There is a correlation between the two, some clusters are more likely to appear in the early days, while others in later days.




