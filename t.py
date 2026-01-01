from misaki import en


if __name__ == "__main__":
    g2p = en.G2P(trf=False, british=False, fallback=None) # no transformer, American English

    text = '[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models.'

    phonemes, tokens = g2p(text)

    print(phonemes) # misˈɑki ɪz ə ʤˈitəpˈi ˈɛnʤən dəzˈInd fɔɹ kˈOkəɹO mˈɑdᵊlz.