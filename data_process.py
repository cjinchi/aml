import pandas as pd
from sklearn.preprocessing import LabelEncoder

weight_map = lambda x: x if pd.isna(x) else float(x[:-3])
sorted_bust_size = ['28a', '28aa', '28b', '28c', '28d', '28dd', '28ddd/e', '28f', '28g', '28h', '28i', '30a', '30aa',
                    '30b', '30c', '30d', '30dd', '30ddd/e', '30f', '30g', '30h', '30i', '32a', '32aa', '32b', '32c',
                    '32d', '32d+', '32dd', '32ddd/e', '32f', '32g', '32h', '32i', '32j', '34a', '34aa', '34b', '34c',
                    '34d', '34d+', '34dd', '34ddd/e', '34f', '34g', '34h', '34i', '34j', '36a', '36aa', '36b', '36c',
                    '36d', '36d+', '36dd', '36ddd/e', '36f', '36g', '36h', '36i', '36j', '38a', '38aa', '38b', '38c',
                    '38d', '38d+', '38dd', '38ddd/e', '38f', '38g', '38h', '38i', '38j', '40b', '40c', '40d', '40dd',
                    '40ddd/e', '40f', '40g', '40h', '40j', '42b', '42c', '42d', '42dd', '42ddd/e', '42f', '42g', '42h',
                    '42j', '44b', '44c', '44d', '44dd', '44ddd/e', '44f', '44g', '44h', '46c', '46ddd/e', '48dd']


# def height_map(x):
#     if pd.isna(x):
#         return x
#     else:
#         # items =
#         # return items[0] * 30.48 + items[1] * 2.54
#         return sum([a * b for a, b in zip([int(i[:-1]) for i in x.split(' ')], [30.48, 2.54])])
height_map = lambda x:x if pd.isna(x) else sum([a * b for a, b in zip([int(i[:-1]) for i in x.split(' ')], [30.48, 2.54])])

def bust_size_map(x):
    if pd.isna(x):
        return x
    else:
        return sorted_bust_size.index(x) + 1


if __name__ == '__main__':
    # print(height_map('5\' 6"'))
    df = pd.read_csv('./data/train.txt')
    # df['weight'] = df['weight'].map(weight_map)
    # df['height'] = df['height'].map(height_map)
    # df['bust size'] = df['bust size'].map(bust_size_map)
    df['label'] = LabelEncoder().fit_transform(df['fit'].astype('category'))
    print(df['label'].head())
    #
    # sub_df_feature = df[['weight', 'size', 'rating', 'bust size', 'height']]
    # sub_df_label = df[['label']]
