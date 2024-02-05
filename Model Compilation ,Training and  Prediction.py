def translate_sentence(sentence, model, english_tokenizer, arabic_tokenizer):
    input_sequence = english_tokenizer.texts_to_sequences([sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=model.input_shape[1], padding='post')
    predicted_sequence = model.predict(input_sequence)

    # Reshape the predicted sequence
    predicted_indices = np.argmax(predicted_sequence, axis=-1)
    predicted_indices = np.squeeze(predicted_indices, axis=0)

    # Convert indices to text
    translated_sentence = arabic_tokenizer.sequences_to_texts([predicted_indices])

    return translated_sentence[0]

new_english_sentence = input()
translated_sentence = translate_sentence(new_english_sentence, model, english_tokenizer, arabic_tokenizer)
print(translated_sentence)
