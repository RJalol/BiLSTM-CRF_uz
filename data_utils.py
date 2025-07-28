def bio_format(text, aspect_terms):
    tokens = text.split()  # oddiy tokenizatsiya
    labels = ['O'] * len(tokens)

    for term in aspect_terms:
        term_tokens = term.split()
        for i in range(len(tokens) - len(term_tokens) + 1):
            if tokens[i:i+len(term_tokens)] == term_tokens:
                labels[i] = 'B-ASP'
                for j in range(1, len(term_tokens)):
                    labels[i+j] = 'I-ASP'
                break
    return tokens, labels
