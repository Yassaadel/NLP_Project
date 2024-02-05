data['english'] = data['english'].apply(str.lower)  # Lowercase English text

english_tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed
english_tokenizer.fit_on_texts(data['english'])
english_vocab_size = len(english_tokenizer.word_index) + 1  # Add 1 for padding token

arabic_tokenizer = Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # Adjust filters for Arabic
arabic_tokenizer.fit_on_texts(data['arabic'])
arabic_vocab_size = len(arabic_tokenizer.word_index) + 1  # Add 1 for padding token

english_sequences = english_tokenizer.texts_to_sequences(data['english'])
arabic_sequences = arabic_tokenizer.texts_to_sequences(data['arabic'])

max_sequence_length = max(len(seq) for seq in english_sequences)
english_padded = pad_sequences(english_sequences, maxlen=max_sequence_length, padding='post')
arabic_padded = pad_sequences(arabic_sequences, maxlen=max_sequence_length, padding='post')
