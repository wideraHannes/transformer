import pytest
import torch

from src.modelling.embedding.positional_encoding import PositionalEncoding


# from src.modelling.positional_encoding import PositionalEncoding

# Define test data
EMBEDDING_DIM = 16
SEQ_LEN = 8
BATCH_SIZE = 2

INPUT = torch.zeros(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)

TEST_DATA = [
    (
        PositionalEncoding(EMBEDDING_DIM, SEQ_LEN),
        INPUT,
        torch.tensor(
            [
                [
                    [
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.8414709568023682,
                        0.5403022766113281,
                        0.3109835982322693,
                        0.9504152536392212,
                        0.0998334139585495,
                        0.9950041770935059,
                        0.031617503613233566,
                        0.9995000958442688,
                        0.009999833069741726,
                        0.9999500513076782,
                        0.0031622713431715965,
                        0.9999950528144836,
                        0.0009999998146668077,
                        0.999999463558197,
                        0.0003162277862429619,
                        0.9999999403953552,
                    ],
                    [
                        0.9092974066734314,
                        -0.416146844625473,
                        0.5911270976066589,
                        0.8065783977508545,
                        0.19866931438446045,
                        0.9800665378570557,
                        0.06320339441299438,
                        0.9980006217956543,
                        0.019998665899038315,
                        0.9997999668121338,
                        0.006324511021375656,
                        0.9999799728393555,
                        0.001999998465180397,
                        0.9999980330467224,
                        0.0006324555142782629,
                        0.9999998211860657,
                    ],
                    [
                        0.14112000167369843,
                        -0.9899924993515015,
                        0.8126488924026489,
                        0.5827536582946777,
                        0.29552018642425537,
                        0.9553365707397461,
                        0.0947260856628418,
                        0.9955034255981445,
                        0.029995499178767204,
                        0.9995500445365906,
                        0.009486687369644642,
                        0.9999549984931946,
                        0.002999995369464159,
                        0.9999955296516418,
                        0.0009486832423135638,
                        0.9999995827674866,
                    ],
                    [
                        -0.756802499294281,
                        -0.6536436080932617,
                        0.9535807371139526,
                        0.30113744735717773,
                        0.3894183337688446,
                        0.9210610389709473,
                        0.1261540651321411,
                        0.9920106530189514,
                        0.03998933359980583,
                        0.9992001056671143,
                        0.012648769654333591,
                        0.9999200105667114,
                        0.00399998901411891,
                        0.9999920129776001,
                        0.001264910795725882,
                        0.9999991655349731,
                    ],
                    [
                        -0.9589242339134216,
                        0.28366217017173767,
                        0.9999465346336365,
                        -0.010342338122427464,
                        0.4794255197048187,
                        0.8775826096534729,
                        0.15745587646961212,
                        0.9875260591506958,
                        0.04997916519641876,
                        0.9987502694129944,
                        0.01581072434782982,
                        0.999875009059906,
                        0.004999978933483362,
                        0.9999874830245972,
                        0.0015811382327228785,
                        0.9999988079071045,
                    ],
                    [
                        -0.279415488243103,
                        0.9601702690124512,
                        0.9471482038497925,
                        -0.320796400308609,
                        0.5646424293518066,
                        0.8253356218338013,
                        0.18860027194023132,
                        0.9820539355278015,
                        0.059964004904031754,
                        0.998200535774231,
                        0.018972521647810936,
                        0.9998200535774231,
                        0.005999963730573654,
                        0.9999819993972778,
                        0.001897365553304553,
                        0.9999982714653015,
                    ],
                    [
                        0.6569865942001343,
                        0.7539022564888,
                        0.8004215955734253,
                        -0.5994374752044678,
                        0.6442176103591919,
                        0.7648422122001648,
                        0.21955609321594238,
                        0.9755998849868774,
                        0.06994284689426422,
                        0.9975509643554688,
                        0.02213412895798683,
                        0.9997549653053284,
                        0.006999942008405924,
                        0.9999755024909973,
                        0.002213592641055584,
                        0.999997615814209,
                    ],
                ],
                [
                    [
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.8414709568023682,
                        0.5403022766113281,
                        0.3109835982322693,
                        0.9504152536392212,
                        0.0998334139585495,
                        0.9950041770935059,
                        0.031617503613233566,
                        0.9995000958442688,
                        0.009999833069741726,
                        0.9999500513076782,
                        0.0031622713431715965,
                        0.9999950528144836,
                        0.0009999998146668077,
                        0.999999463558197,
                        0.0003162277862429619,
                        0.9999999403953552,
                    ],
                    [
                        0.9092974066734314,
                        -0.416146844625473,
                        0.5911270976066589,
                        0.8065783977508545,
                        0.19866931438446045,
                        0.9800665378570557,
                        0.06320339441299438,
                        0.9980006217956543,
                        0.019998665899038315,
                        0.9997999668121338,
                        0.006324511021375656,
                        0.9999799728393555,
                        0.001999998465180397,
                        0.9999980330467224,
                        0.0006324555142782629,
                        0.9999998211860657,
                    ],
                    [
                        0.14112000167369843,
                        -0.9899924993515015,
                        0.8126488924026489,
                        0.5827536582946777,
                        0.29552018642425537,
                        0.9553365707397461,
                        0.0947260856628418,
                        0.9955034255981445,
                        0.029995499178767204,
                        0.9995500445365906,
                        0.009486687369644642,
                        0.9999549984931946,
                        0.002999995369464159,
                        0.9999955296516418,
                        0.0009486832423135638,
                        0.9999995827674866,
                    ],
                    [
                        -0.756802499294281,
                        -0.6536436080932617,
                        0.9535807371139526,
                        0.30113744735717773,
                        0.3894183337688446,
                        0.9210610389709473,
                        0.1261540651321411,
                        0.9920106530189514,
                        0.03998933359980583,
                        0.9992001056671143,
                        0.012648769654333591,
                        0.9999200105667114,
                        0.00399998901411891,
                        0.9999920129776001,
                        0.001264910795725882,
                        0.9999991655349731,
                    ],
                    [
                        -0.9589242339134216,
                        0.28366217017173767,
                        0.9999465346336365,
                        -0.010342338122427464,
                        0.4794255197048187,
                        0.8775826096534729,
                        0.15745587646961212,
                        0.9875260591506958,
                        0.04997916519641876,
                        0.9987502694129944,
                        0.01581072434782982,
                        0.999875009059906,
                        0.004999978933483362,
                        0.9999874830245972,
                        0.0015811382327228785,
                        0.9999988079071045,
                    ],
                    [
                        -0.279415488243103,
                        0.9601702690124512,
                        0.9471482038497925,
                        -0.320796400308609,
                        0.5646424293518066,
                        0.8253356218338013,
                        0.18860027194023132,
                        0.9820539355278015,
                        0.059964004904031754,
                        0.998200535774231,
                        0.018972521647810936,
                        0.9998200535774231,
                        0.005999963730573654,
                        0.9999819993972778,
                        0.001897365553304553,
                        0.9999982714653015,
                    ],
                    [
                        0.6569865942001343,
                        0.7539022564888,
                        0.8004215955734253,
                        -0.5994374752044678,
                        0.6442176103591919,
                        0.7648422122001648,
                        0.21955609321594238,
                        0.9755998849868774,
                        0.06994284689426422,
                        0.9975509643554688,
                        0.02213412895798683,
                        0.9997549653053284,
                        0.006999942008405924,
                        0.9999755024909973,
                        0.002213592641055584,
                        0.999997615814209,
                    ],
                ],
            ]
        ),
    )
]


# Positional Encoding Layer Tests
@pytest.mark.parametrize(
    "encoding_layer, input, expected", TEST_DATA, ids=["positional_encoding"]
)
def test_attention(encoding_layer, input, expected):
    """Test the Positional Encoding layer."""
    actual = encoding_layer(input)
    print("Actual Output:", actual)
    print("Expected Output:", expected)
    assert torch.allclose(encoding_layer(input), expected)
