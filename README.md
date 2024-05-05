# ***Image Caption Generator***

Image captioning is the process of generating textual descriptions for images
automatically. It combines computer vision and natural language processing techniques to
understand the content of an image and generate a coherent and relevant description.

## CNN & RNN (LSTM)

To perform Image Captioning we will require two deep learning models combined into one for the training purpose. <br>
CNNs extract the features from the image of some vector size aka the vector embeddings. The size of these embeddings depend on the type of pretrained network being used for the feature extraction
LSTMs are used for the text generation process. The image embeddings are concatenated with the word embeddings and passed to the LSTM to generate the next word

## Dataset

A new benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the entities and events. <br>
The images don't contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations. <br>

Kaggle: [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) <br>
Hugging Face: [Flickr 8k Dataset](https://huggingface.co/datasets/jxie/flickr8k)

<hr>
