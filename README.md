# ***Image Caption Generator***

Image captioning is the process of generating textual descriptions for images
automatically. It combines computer vision and natural language processing techniques to
understand the content of an image and generate a coherent and relevant description.

## CNN & RNN (LSTM)

To perform Image Captioning we will require two deep learning models combined into one for the training purpose. <br>
CNNs extract the features from the image of some vector size aka the vector embeddings. The size of these embeddings depend on the type of pretrained network being used for the feature extraction
LSTMs are used for the text generation process. The image embeddings are concatenated with the word embeddings and passed to the LSTM to generate the next word <br>

![image](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/9a0cdfca-e1f0-4284-b54d-6566b7776b3e)


## Dataset

A new benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the entities and events. <br>
The images don't contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations. <br>

Kaggle: [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) <br>
Hugging Face: [Flickr 8k Dataset](https://huggingface.co/datasets/jxie/flickr8k)

<hr>

## Notebook

Link to Kaggle notebook: [Image-Caption-Generator_CNN-LSTM (PyTorch)](https://www.kaggle.com/code/youssefaboelwafa/image-caption-generator-cnn-lstm-pytorch)

<hr>

## Caption Generation (Good Examples)
![output_37_3](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/e105bebb-94fc-4a84-bc9e-a2c678e7936a) <br>
![output_39_2](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/ec2d7cd4-f258-409e-90cb-1e707d391168) <br>
![output_39_9](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/01e4523a-b31e-469d-8de0-2deb9c3b096b) <br>
![output_37_1](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/03f75b9a-82f8-451c-afef-15cf6e244266) <br>

<hr>

## Caption Generation (Bad Examples)
![output_37_2](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/a874e8dc-c9e6-4e2f-a16f-5c85468cd016) <br>
![output_39_8](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/ab5f8e24-d663-479e-95df-cf8c45f7e030) <br>
![output_39_7](https://github.com/YoussefAboelwafa/Image-Caption-Generator/assets/96186143/f970ee0e-8bd6-421b-942d-6c0f796e9bac) <br>





