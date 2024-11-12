# marine-microorganisms-clustering

Abstract: The objective in this analysis is to group microorganisms according to their type by utilizing state-ofthe-art clustering algorithms. This will allow us to better understand the different species in the dataset and the various characteristics that define them. The dataset comprises more than 1 million unannotated images gathered from the ocean by marine biologists associated with the Weizmann Institute. We will also closely examine the data’s temporal trends to identify patterns that may emerge over time. In addition to our primary objectives, we will perform various secondary analyses to explore the dataset further. This may include visualizing the data in different ways to gain new perspectives or utilizing machine learning techniques to identify complex patterns that may be difficult to discern through traditional analysis methods.

2.1 Data Pre-Processing
Data analysis and modeling processes require accurate and high-quality input data. In the case of image analysis, it is essential to ensure that the images being used are of adequate size and resolution for the task at hand. The raw data for this particular project consists of a single image comprising several cells. In order to prepare this data for analysis, our first step is to split each image into individual images, each containing only a single organism. This will allow us to isolate and analyze each cell separately. Moreover, the images are relatively small with various sizes, for example, some images contain only 20Π20 pixels. This is inadequate for a pre-trained vision model to analyze the cells in the image accurately. Therefore, we changed the size of each image to be 64Π64 pixels using standard image transformations provided by Pytorch [7]. This will ensure that the cells are large enough for the model to detect features and patterns accurately. 2.2 Extract Meaningful Representations We leveraged a pre-trained DINO model [1] that was trained on ImageNet [3] to derive rich and meaningful image representations. Since the distribution of the images that were used to train this model is significantly different than our data, we inspect the attention maps (Figure 3) obtained by the pre-trained model on our data. To enhance the model’s performance we additionally fine-tuned this model on our specific dataset, for one epoch. Thus we extracted the most informative features relevant to our images.

![image](https://github.com/user-attachments/assets/c6270015-42bf-463a-b7d7-e01047e98f7f)
