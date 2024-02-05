model = Sequential([
   Embedding(english_vocab_size, 128, input_length=max_sequence_length, mask_zero=True),
   Bidirectional(LSTM(512, return_sequences=True)),
   Dropout(0.2),
   Bidirectional(LSTM(512)),
   RepeatVector(max_sequence_length),
   Bidirectional(LSTM(512, return_sequences=True)),
   Dropout(0.2),
   TimeDistributed(Dense(arabic_vocab_size, activation='softmax'))
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
