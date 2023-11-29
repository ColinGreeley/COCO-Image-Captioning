import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers, metrics, losses, applications
import tensorflow_addons as tfa
import collections
import math
import os
import json
import random
from PIL import Image
from tokenizers import BertWordPieceTokenizer
import tensorflow.keras.backend as K
import gc



def get_data():
    trainval_image_dir = "D:/Data/COCO/train2014/train2014/"
    trainval_captions_dir = "D:/Data/COCO/annotations_trainval2014/annotations"
    test_image_dir = "D:/Data/COCO/val2017/val2017/"
    test_captions_dir = "D:/Data/COCO/annotations_trainval2017/annotations"
    trainval_captions_filepath = os.path.join(trainval_captions_dir, 'captions_train2014.json')
    test_captions_filepath = os.path.join(test_captions_dir, 'captions_val2017.json')
    all_filepaths = np.array([os.path.join(trainval_image_dir, f) for f in os.listdir(trainval_image_dir)])
    rand_indices = np.arange(len(all_filepaths))
    np.random.shuffle(rand_indices)
    split = int(len(all_filepaths)*0.8)
    train_filepaths, valid_filepaths = all_filepaths[rand_indices[:split]], all_filepaths[rand_indices[split:]] 
    print(f"Train dataset size: {len(train_filepaths)}")
    print(f"Valid dataset size: {len(valid_filepaths)}")
    with open(trainval_captions_filepath, 'r') as f:
        trainval_data = json.load(f)
    trainval_captions_df = pd.json_normalize(trainval_data, "annotations")
    trainval_captions_df["image_filepath"] = trainval_captions_df["image_id"].apply(lambda x: os.path.join(trainval_image_dir, 'COCO_train2014_'+format(x, '012d')+'.jpg'))
    train_captions_df = trainval_captions_df[trainval_captions_df["image_filepath"].isin(train_filepaths)]
    train_captions_df = preprocess_captions(train_captions_df)
    valid_captions_df = trainval_captions_df[trainval_captions_df["image_filepath"].isin(valid_filepaths)]
    valid_captions_df = preprocess_captions(valid_captions_df)
    with open(test_captions_filepath, 'r') as f:
        test_data = json.load(f)
    test_captions_df = pd.json_normalize(test_data, "annotations")
    test_captions_df["image_filepath"] = test_captions_df["image_id"].apply(lambda x: os.path.join(test_image_dir, format(x, '012d')+'.jpg'))
    test_captions_df = preprocess_captions(test_captions_df)
    return train_captions_df, valid_captions_df, test_captions_df
    
    
def preprocess_captions(image_captions_df):
    """ Preprocessing the captions """
    image_captions_df["preprocessed_caption"] = "[START] " + image_captions_df["caption"].str.lower().str.replace('[^\w\s]','') + " [END]"
    return image_captions_df

@tf.function
def parse_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256], antialias=True)
    #image = image * 2.0 - 1.0
    image = image * 255
    return image   
 
def generate_tokenizer(captions_df, n_vocab):
    """ Generate the tokenizer with given captions """
    # Define the tokenizer
    tokenizer = BertWordPieceTokenizer(unk_token="[UNK]", clean_text=True, lowercase=True,)
    # Train the tokenizer
    tokenizer.train_from_iterator(captions_df["preprocessed_caption"].tolist(), vocab_size=n_vocab, special_tokens=["[PAD]", "[UNK]", "[START]", "[END]"])
    return tokenizer   
    
    
def generate_tf_dataset(image_captions_df, tokenizer=None, n_vocab=8000, pad_length=33, batch_size=32, training=False):
    """ Generate the tf.data.Dataset"""
    # If the tokenizer is not available, create one
    if not tokenizer:
        tokenizer = generate_tokenizer(image_captions_df, n_vocab)
    # Get the caption IDs using the tokenizer
    image_captions_df["caption_token_ids"] = [enc.ids for enc in tokenizer.encode_batch(image_captions_df["preprocessed_caption"])]
    vocab = tokenizer.get_vocab()
    # Add the padding to short sentences and truncate long ones
    image_captions_df["caption_token_ids"] = image_captions_df["caption_token_ids"].apply(lambda x: x+[vocab["[PAD]"]]*(pad_length - len(x) + 2) if pad_length + 2 >= len(x) else x[:pad_length + 1] + [x[-1]]) 
    # Create a dataset with images and captions
    dataset = tf.data.Dataset.from_tensor_slices({"image_filepath": image_captions_df["image_filepath"], "caption_token_ids": np.array(image_captions_df["caption_token_ids"].tolist())})
    # Each sample in our dataset consists of
    # (image, caption token IDs, position IDs), (caption token IDs offset by 1)
    #valid = np.zeros((batch_size, 1))
    dataset = dataset.map(lambda x: ((parse_image(x["image_filepath"]), x["caption_token_ids"][:-1]), x["caption_token_ids"][1:]))
    
    # Shuffle and batch data in the training mode
    if training:
        dataset = dataset.shuffle(buffer_size=batch_size*100)
    dataset = dataset.batch(batch_size)
    return dataset, tokenizer

def data_gen(image_captions_df, tokenizer, batch_size=25, pad_length=33):
    image_captions_df["caption_token_ids"] = [enc.ids for enc in tokenizer.encode_batch(image_captions_df["preprocessed_caption"])]
    vocab = tokenizer.get_vocab()
    image_captions_df["caption_token_ids"] = image_captions_df["caption_token_ids"].apply(lambda x: x+[vocab["[PAD]"]]*(pad_length - len(x) + 2) if pad_length + 2 >= len(x) else x[:pad_length + 1] + [x[-1]]) 
    captions = np.array(image_captions_df["caption_token_ids"].tolist())
    images = image_captions_df["image_filepath"].to_numpy()
    #images = np.array([parse_image(im, image_size, image_size) for im in tqdm(image_captions_df["image_filepath"])])
    #print("Data len:", len(images))
    gc.collect()
    while True:
        idxs = np.random.randint(0, len(image_captions_df["caption_token_ids"]), batch_size)
        img_batch = images[idxs]
        #img_batch = np.array([parse_image(im) for im in img_batch])
        img_batch = tf.map_fn(parse_image, img_batch, dtype=tf.float32)
        caption = captions[idxs]
        #print(caption.shape)
        yield (img_batch, caption[:,:-1]), caption[:,1:]
    
def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

      Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.

      Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.

      Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.

      Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches
        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                   precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                             possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0
        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0
        ratio = float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)
        bleu = geo_mean * bp
        return (bleu, precisions, bp, ratio, translation_length, reference_length)
    
class BLEUMetric(object):
    
    def __init__(self, tokenizer, name='bleu_metric', **kwargs):
        """ Computes the BLEU score (Metric for machine translation) """
        super().__init__()
        self.tokenizer = tokenizer
    
      #self.vocab = vocabulary
      #self.id_to_token_layer = StringLookup(vocabulary=self.vocab, num_oov_indices=0, oov_token='[UNKUNK]', invert=True)
    
    def calculate_bleu_from_predictions(self, real, pred):
        """ Calculate the BLEU score for targets and predictions """
        # Get the predicted token IDs
        pred_argmax = tf.argmax(pred, axis=-1)  
        # Convert token IDs to words using the vocabulary and the StringLookup
        pred_tokens = np.array([[self.tokenizer.id_to_token(pp) for pp in p] for p in pred_argmax])
        real_tokens = tf.constant([[self.tokenizer.id_to_token(rr) for rr in r] for r in real])
        
        def clean_text(tokens):
            """ Clean padding and other tokens to only keep meaningful words """
            # 3. Strip the string of any extra white spaces
            translations_in_bytes = tf.strings.strip(
                        # 2. Replace everything after the eos token with blank
                        tf.strings.regex_replace(
                            # 1. Join all the tokens to one string in each sequence
                            tf.strings.join(
                                tf.transpose(tokens), separator=' '
                            ),
                        "\[END\].*", ""),
                   )
            # Decode the byte stream to a string
            translations = np.char.decode( #
                translations_in_bytes.numpy().astype(np.bytes_), encoding='utf-8'
            )
            # If the string is empty, add a [UNK] token
            # Otherwise get a Division by zero error
            translations = [sent if len(sent)>0 else "[UNK]" for sent in translations ]
            # Split the sequences to individual tokens 
            translations = np.char.split(translations).tolist()
            return translations
        
        # Get the clean versions of the predictions and real seuqences
        pred_tokens = clean_text(pred_tokens)
        # We have to wrap each real sequence in a list to make use of a function to compute bleu
        real_tokens = [[token_seq] for token_seq in clean_text(real_tokens)]
        # The compute_bleu method accpets the translations and references in the following format
        # tranlation - list of list of tokens
        # references - list of list of list of tokens
        bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu(real_tokens, pred_tokens, smooth=False)
        return bleu
    
def CNNForward(x_in, k=3, dropout=0.25, norm=True):
    if norm:
        x = layers.LayerNormalization()(x_in)
    else:
        x = x_in
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.SeparableConv2D(x_in.shape[-1]*2, k, padding='same', activation='swish', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = layers.Conv2D(x_in.shape[-1], 1, padding='same')(x)
    return layers.Add()([x_in, x])
   
def PosEmbedding(latent_dim):
    def pos_embedding(x):
        t_embed = tf.range(100)[tf.newaxis,:tf.shape(x)[1]]
        t_embed = tf.tile(t_embed, [tf.shape(t_embed)[0],1]) # (batch_size, num_people, 64)
        t_embed = layers.Embedding(100, latent_dim)(t_embed)
        x = layers.Add()([x, t_embed])
        return x
    return pos_embedding
    
def SelfAttention(heads, size, dropout, use_causal_mask=False, return_res=False):
    def self_attention(x_in):
        x = layers.LayerNormalization()(x_in)
        x = layers.Activation('swish')(x)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=size, dropout=dropout)(x, x, x, use_causal_mask=use_causal_mask)
        if return_res:
            return x
        return layers.Add()([x_in, x])
    return self_attention

def CrossAttention(heads, size, dropout, return_res=False):
    def cross_attention(x_in, y_in):
        x = layers.LayerNormalization()(x_in)
        x = layers.Activation('swish')(x)
        y = layers.LayerNormalization()(y_in)
        y = layers.Activation('swish')(y)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=size, dropout=dropout)(query=x, value=y)
        if return_res:
            return x
        return layers.Add()([x_in, x])
    return cross_attention   

def SqueezeExcite(x_in, r=4):
    filters = x_in.shape[-1]
    #se = layers.LSTM(filters//r, return_sequences=False)(x_in)
    if len(x_in.shape) == 4:
        se = layers.GlobalAveragePooling2D()(x_in)
    elif len(x_in.shape) == 3:
        se = layers.GlobalAveragePooling1D()(x_in)
    se = layers.Dense(filters//r, activation='swish', kernel_initializer='he_normal')(se) # , kernel_initializer='he_normal'
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.multiply([x_in, se])

def FeedForward(activation='swish', dropout=0.25, r=1, return_res=False):
    def feed_forward(x_in):
        in_shape = x_in.shape[-1]
        x = layers.LayerNormalization()(x_in)
        x = layers.Activation('swish')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(int(in_shape*r), activation=activation, kernel_initializer='he_normal')(x)
        x = layers.Dense(in_shape)(x)
        if return_res:
            return x
        return layers.Add()([x_in, x])
    return feed_forward
 
def image_encoder(image_input):
    #x = layers.BatchNormalization()(image_input)
    x = layers.Conv2D(32, 5, padding='same', activation='swish', kernel_initializer='he_normal')(image_input)
    x = layers.Conv2D(32, 3, strides=2, padding='same')(x) # (128, 128)
    x = CNNForward(x)
    x = CNNForward(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x) # (64, 64)
    x = CNNForward(x)
    x = CNNForward(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x) # (32, 32)
    x = CNNForward(x)
    x = CNNForward(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x) # (16, 16)
    x = CNNForward(x)
    x = CNNForward(x)
    x = layers.Conv2D(384, 3, strides=2, padding='same')(x) # (8, 8)
    x = layers.LayerNormalization()(x)
    return x
 
def transformer(x, y, d_model, depth, heads=4, dropout=0.25):
    x_shape = x.shape
    x = layers.Reshape((x_shape[1]*x_shape[2],x_shape[3]))(x)
    x = PosEmbedding(x_shape[3])(x)
    y = PosEmbedding(d_model)(y)
    for _ in range(depth):
        x = SelfAttention(heads, x.shape[-1]//2, dropout=dropout)(x)
        x = FeedForward(dropout=dropout, r=1)(x)
        y = SelfAttention(heads, d_model, dropout=dropout, use_causal_mask=True)(y)
        y = CrossAttention(heads, d_model, dropout=dropout)(y, x)
        y = FeedForward(dropout=dropout, r=1)(y)
    x = layers.Reshape((x_shape[1], x_shape[2],x_shape[3]))(x)
    return x, y
    
def freeze_model(m, block):
    m.trainable = True
    i = 0
    while i < len(m.layers):
        if 'block{}'.format(block) in m.layers[i].name:
            break
        m.layers[i].trainable = False
        i += 1
    while i < len(m.layers):
        if isinstance(m.layers[i], layers.BatchNormalization):
            m.layers[i].trainable = False
        i += 1
    return m

def make_model(tokenizer, d_model=786):
    image_input = tf.keras.layers.Input(shape=(256, 256, 3))
    caption_input = tf.keras.layers.Input(shape=(None,))
    # Token embeddings
    input_embedding = tf.keras.layers.Embedding(len(tokenizer.get_vocab()), d_model, mask_zero=True)
    embed_out = input_embedding(caption_input)
    # Position embeddings
    # Combined token position embeddings
    #image_features = image_encoder(image_input)
    encoder = applications.efficientnet_v2.EfficientNetV2B2(include_top=False, input_shape=(256,256,3))
    #encoder.summary()
    encoder = freeze_model(encoder, 8)
    x = encoder(image_input)
    x = layers.Conv2D(x.shape[-1], 1)(x)
    x, embed_out = transformer(x, embed_out, d_model, depth=2)
    """
    x = image_input
    for width in [64, 128, 256, 512, 1024]:
        x = layers.Conv2D(width, 3, strides=2, padding='same')(x) # (128, 128)
        x = CNNForward(x)
        x = CNNForward(x)
        if width >= 1024:
            x, embed_out = transformer(x, embed_out, d_model, 4)
    """
    # Final prediction layer
    embed_out = layers.LayerNormalization()(embed_out)
    embed_out = layers.Activation('swish')(embed_out)
    embed_out = layers.Dropout(0.25)(embed_out)
    caption_out = layers.Dense(len(tokenizer.get_vocab()), activation='softmax', name='caption')(embed_out)
    # Define the final model and compile
    model = tf.keras.models.Model(inputs=[image_input, caption_input], outputs=caption_out)
    model.summary()
    return model

def generate_caption(model, image_input, tokenizer, n_samples):
    # 2 -> [START]
    batch_tokens = start_tokens = np.repeat(np.array([[2]]), n_samples, axis=0)
    for i in range(30):
        if np.all(batch_tokens[:,-1] == 3):
            break
        #print(image_input.shape, batch_tokens.shape)
        probs = model((image_input, batch_tokens)).numpy()
        #batch_tokens = np.argmax(probs, axis=-1)
        batch_tokens = np.concatenate([batch_tokens, np.argmax(probs, axis=-1)[:,-1][:,np.newaxis]], axis=-1)
    predicted_text = []
    for sample_tokens in batch_tokens:
        sample_predicted_token_ids = sample_tokens.ravel()
        sample_predicted_tokens = []
        for wid in sample_predicted_token_ids[1:]:
            if wid == 3:
                break
            sample_predicted_tokens.append(tokenizer.id_to_token(wid))
        sample_predicted_text = " ".join([tok for tok in sample_predicted_tokens])
        sample_predicted_text = sample_predicted_text.replace(" ##", "")
        predicted_text.append(sample_predicted_text)
    return predicted_text

def gen_captions(model, df, e):
    n_samples = 5
    test_dataset, _ = generate_tf_dataset(df.sample(n=n_samples), tokenizer=tokenizer, n_vocab=n_vocab, batch_size=n_samples, training=False)
    for batch in test_dataset.take(1):
        (batch_image_input, batch_caption), batch_true_caption = batch
    batch_predicted_text = generate_caption(model, batch_image_input, tokenizer, n_samples)
    fig, axes = plt.subplots(n_samples, 2, figsize=(15,30))
    for i,(sample_image_input, sample_true_caption, sample_predicated_caption) in enumerate(zip(batch_image_input, batch_true_caption, batch_predicted_text)):
        sample_true_caption_tokens  = [tokenizer.id_to_token(wid) for wid in sample_true_caption.numpy().ravel()]
        sample_true_text = []
        for tok in sample_true_caption_tokens:
            if tok == '[END]':
                break
            sample_true_text.append(tok)
        sample_true_text = " ".join(sample_true_text).replace(" ##", "")
        axes[i][0].imshow(np.clip(sample_image_input.numpy()/255.0, 0, 1))
        axes[i][0].axis('off')
        true_annotation = f"TRUE: {sample_true_text}"
        predicted_annotation = f"PRED: {sample_predicated_caption}"
        axes[i][1].text(0, 0.75, true_annotation, fontsize=10)
        axes[i][1].text(0, 0.25, predicted_annotation, fontsize=10)
        axes[i][1].axis('off')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(f"captions/caption_{e}.jpg")
    plt.close()
    
        


if __name__ == '__main__':
    
    train_captions_df, valid_captions_df, test_captions_df = get_data()
    
    n_vocab=8000
    tokenizer = generate_tokenizer(train_captions_df, n_vocab=n_vocab)
    bleu_metric = BLEUMetric(tokenizer=tokenizer)
    optimizer = optimizers.Nadam(learning_rate=3e-4)
    #optimizer = optimizers.RMSprop(0.00005)
    model = make_model(tokenizer)
    #model = tf.keras.models.load_model("./caption.keras")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[metrics.SparseCategoricalAccuracy()])

    batch_size=26
    steps_per_epoch = 5000

    train_fraction = 1.0
    valid_fraction = 0.2


    for e in range(1000):
        print(f"\nEpoch: {e+1}")
        
        #train_dataset, _ = generate_tf_dataset(train_captions_df.sample(frac=train_fraction), tokenizer=tokenizer, n_vocab=n_vocab, batch_size=batch_size, training=True)
        #valid_dataset, _ = generate_tf_dataset(valid_captions_df.sample(frac=valid_fraction), tokenizer=tokenizer, n_vocab=n_vocab, batch_size=batch_size, training=False)
        train_dataset = data_gen(train_captions_df, tokenizer=tokenizer, batch_size=batch_size)
        val_dataset = data_gen(valid_captions_df, tokenizer=tokenizer, batch_size=batch_size)
        
        loss_list, acc_list = [], []
        for i in range(steps_per_epoch):
            v_batch = next(train_dataset)
            loss, acc = model.train_on_batch(v_batch[0], v_batch[1])
            loss_list.append(loss)
            acc_list.append(acc)
            loss, acc = str(round(np.mean(loss_list), 4)),  str(round(np.mean(acc_list), 4))
            print(f"Step: {i+1}/{steps_per_epoch} | Loss: {loss} | Acc: {acc}", end='\r')
        model.save("./caption.keras")
        gen_captions(model, test_captions_df, e)
        
        valid_loss, valid_accuracy, valid_bleu = [], [], []
        for i in range(200):
            v_batch = next(val_dataset)
            #print(f"{i+1} batches processed", end='\r')
            loss, accuracy = model.test_on_batch(v_batch[0], v_batch[1])
            batch_predicted = model(v_batch[0])
            bleu_score = bleu_metric.calculate_bleu_from_predictions(v_batch[1], batch_predicted)
            valid_loss.append(loss)
            valid_accuracy.append(accuracy)
            valid_bleu.append(bleu_score)
            
        print(f"\nvalid_loss: {round(np.mean(valid_loss), 4)} - valid_accuracy: {round(np.mean(valid_accuracy), 4)} - valid_bleu: {round(np.mean(valid_bleu), 4)}")
            
            