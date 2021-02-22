# Libs and packages
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
import matplotlib.pyplot as plt


# Functions
def vectorize_stories(data, word_index, max_story_len, max_question_len):
    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []

    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


if __name__ == '__main__':

    # Step-1: Reading test and train data sets
    with open("train_qa.txt", "rb") as fp:  # Unpickling
        train_data = pickle.load(fp)

    with open("test_qa.txt", "rb") as fp:  # Unpickling
        test_data = pickle.load(fp)

    # Step-2: Creating vocab set and max len for story and que
    vocab = set()
    all_data = test_data + train_data
    for story, question, answer in all_data:
        vocab = vocab.union(set(story))
        vocab = vocab.union(set(question))
    vocab.add('no')
    vocab.add('yes')
    print(vocab)

    vocab_len = len(vocab) + 1
    vocab_size = len(vocab) + 1
    max_story_len = max([len(data[0]) for data in all_data])
    max_question_len = max([len(data[1]) for data in all_data])
    print(max_story_len, max_question_len)

    # Step-3: vocab tokenization and data vectorization
    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)
    print(tokenizer.word_index)

    inputs_train, queries_train, answers_train = vectorize_stories(train_data, tokenizer.word_index, max_story_len, max_question_len)
    inputs_test, queries_test, answers_test = vectorize_stories(test_data, tokenizer.word_index, max_story_len, max_question_len)
    input_sequence = Input((max_story_len,))
    question = Input((max_question_len,))

    # Step-4: Model creation
    # Input gets embedded to a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
    input_encoder_m.add(Dropout(0.3))
    # This encoder will output:
    # (samples, story_maxlen, embedding_dim)

    # embed the input into a sequence of vectors of size query_maxlen
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
    input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)

    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=64,
                                   input_length=max_question_len))
    question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # shape: `(samples, story_maxlen, query_maxlen)`
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

    # concatenate the match matrix with the question vector sequence
    answer = concatenate([response, question_encoded])
    print(answer)
    # Reduce with RNN (LSTM)
    answer = LSTM(32)(answer)  # (samples, 32)

    # Regularization with Dropout
    answer = Dropout(0.5)(answer)
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)

    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

    # build the final model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Step-5: Training the model
    train = 1
    run_epochs = 150
    filename = 'chatbot_150_epochs.h5'
    if train == 1:
        history = model.fit([inputs_train, queries_train], answers_train, batch_size=32, epochs=run_epochs,
                            validation_data=([inputs_test, queries_test], answers_test))
        model.save(filename)
        # %matplotlib inline
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    # Step-6: predicting using the model
    model.load_weights(filename)
    K = 15
    pred_results = model.predict(([inputs_test, queries_test]))
    story = ' '.join(word for word in test_data[K][0])
    print(story)
    query = ' '.join(word for word in test_data[K][1])
    print(query)
    print("True Test Answer from Data is:", test_data[K][2])
    # Generate prediction from model
    #print(pred_results[K])
    val_max = np.argmax(pred_results[K])

    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key

    print("Predicted answer is: ", k)
    print("Probability of certainty was: ", pred_results[K][val_max])