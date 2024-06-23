from probables import BloomFilter


def apply_bloomfilter(data_column, false_positive_rate):
    bloom_filter = BloomFilter(est_elements=len(data_column), false_positive_rate=false_positive_rate)

    for element in data_column:
        bloom_filter.add(str(element))

    # Bloom filter where we act like id is a unique six letter name
    # name_bloom_filter = apply_bloomfilter(features["Unnamed: 0"], 0.4)
    # name_tester = "999999"
    # if name_bloom_filter.check(name_tester):
    #    print(f"{name_tester} is probably in the set.")
    # else:
    #    print(f"{name_tester} is definitely not in the set.")

    return bloom_filter
