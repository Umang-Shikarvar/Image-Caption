import torch
from torchvision import transforms
from PIL import Image
import pickle

from model import Encoder, Decoder
from dataloader import Vocabulary


# Configuration
EMBED_SIZE = 256
HIDDEN_SIZE = 512
CHECKPOINT_DIR = "checkpoints"
ENCODER_PATH = f"{CHECKPOINT_DIR}/encoder_epoch_5.pth"
DECODER_PATH = f"{CHECKPOINT_DIR}/decoder_epoch_5.pth"
VOCAB_PATH = "vocab.pkl"


# Load vocabulary
with open(VOCAB_PATH, 'rb') as f:
    vocab_dict = pickle.load(f)

vocab = Vocabulary(vocab_dict["word2idx"], vocab_dict["idx2word"])
VOCAB_SIZE = len(vocab)

# Load models
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

encoder = Encoder(EMBED_SIZE).to(device)
decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(device)

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))

encoder.eval()
decoder.eval()


# Image preprocessing
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


# Function to clean the output sentence
def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0 or i==2:  # Start and unknown token
            continue
        if i == 1:          # End of sentence token
            break
        sentence = sentence + " " + word
    return sentence


# Function to generate caption for an image
def generate_caption(test_image: Image.Image) -> str:
    # Preprocess the test image
    test_image = transform_test(test_image).unsqueeze(0)  # Add batch dimension

    # Move the preprocessed image to the appropriate device
    test_image = test_image.to(device)

    # Pass the test image through the encoder
    with torch.no_grad():
        features = encoder(test_image).unsqueeze(1)

    # Generate captions with the decoder
    with torch.no_grad():
        output = decoder.sample(features)

    # Convert the output into a clean sentence
    caption = clean_sentence(output, vocab.idx2word)
    return caption