{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Predictive Model for Breast Cancer Incidence\n",
        "Bharat Solanky and Aadya Chawla\n",
        "\n",
        "[Link to GitHub page ](https://bharatsolanky.github.io/)\n",
        "## Project Goals:\n",
        "\n",
        "The goal of this project is to produce a predictive model to determine likelihood of breast cancer incidence in female patients using physical metrics, such as concavity, radius, and texture of perceived benign tumors. \n",
        "\n",
        "## Project DataSet and Plan:\n",
        "\n",
        "The datasets we plan to work with are from Kaggle.com [1](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)\n",
        "[2](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer)\n",
        "[3](https://www.kaggle.com/datasets/0248260fceaaaab93ceb231f0deb49f979a9ce4ed30f54260c8a18d9270bbcb0)\n",
        ". As a background, the medical journals report that the accuracy of visually diagnosed breast FNA is about 94.3% with mean sensitivity of 91 percent and specificity of 87%.\n",
        "The dataset will be randomly divided into two disjoint subgroups, first to train the prediction model and other for testing the accuracy of the developed models. \n",
        "Based on  preliminary analysis, the three features which are highly associated with the diagnosis of breast cancer are the following (the description of variables is below):\n",
        "\n",
        "1. Concave_points_worst: The average value for cancer is 0.18, whereas it is 0.07 for benign;\n",
        "2. Radius_worst: The average value for cancer is 21.13, whereas it is 13.38 for benign; and\n",
        "3. Texture_worst: The average value for cancer is 29.32 whereas it is 23.52 for benign;\n",
        "\n",
        "Our goal is to also provide guidelines for medical professionals to assist them with the process of visual diagnosis (prediction) by identifying potentially extreme values above/below which the likelihood of breast cancer changes significantly. For example, for what value of  Radius_worst does the possibility of cancer have a likelihood of 95% or 100%. \n",
        "\n",
        "\n",
        "## Collaboration Plan:\n",
        "We have created a google colab drive to share files and so that we can easily edit and write code live. We will meet weekly leading up to the final deadline to divide the work and enhance our project objectives. Currently, we use text message to communicate with each other. \n",
        "\n",
        "## ETL (Extraction, Transform, and Load):\n",
        "We loaded three datasets which are all .csv files available on the Kaggle.com website. \n",
        "The first dataset has information of 32 features described above for 569 patients. \n",
        "The second dataset is for confirmed malignant tumor patients, both currently alive and dead, and has data for 4,534 patients, including tumor stage, tumor size, and patient hormone levels.\n",
        "The third dataset is for confirmed malignant tumor patients, both currently alive and dead, and has data for 317 patients, including tumor stage and type of surgery required\n",
        "\n",
        "The first download does not have any missing data for any of the features.\n",
        "\n",
        "\n",
        "###Filtering, Sorting, and Plotting Data Points\n",
        "Below we have filtered data by malignant patient type, sorted to only show concavity points of their cells, and plotted it as a histogram.\n"
      ],
      "metadata": {
        "id": "wJSpOWFf_lSf"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "95uTqFnRCoMp",
        "outputId": "ab81334f-7960-4588-e376-387606c98046"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "#mounting google collab to google drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "! pwd\n",
        "\n",
        "#directory contains csv file\n",
        "%cd /content/drive/My Drive/DataScienceProject\n",
        "!git pull\n",
        "\n",
        "\n",
        "import pandas as pd\n",
        "from matplotlib import pyplot as plt\n",
        "from itertools import cycle, islice\n",
        "pd.options.display.max_rows = 8\n",
        "\n",
        "df = pd.read_csv(\"data.csv\") \n",
        "#This reads the data file which is named data.csv\n",
        "\n",
        "\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 388
        },
        "id": "srTOGolaDERQ",
        "outputId": "086d49e0-97cd-4688-c930-cf0dcad15f0c"
      },
      "execution_count": 90,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/DataScienceProject\n",
            "/content/drive/My Drive/DataScienceProject\n",
            "fatal: not a git repository (or any parent up to mount point /content)\n",
            "Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "0    842302         M        17.99         10.38          122.80     1001.0   \n",
              "1    842517         M        20.57         17.77          132.90     1326.0   \n",
              "2  84300903         M        19.69         21.25          130.00     1203.0   \n",
              "3  84348301         M        11.42         20.38           77.58      386.1   \n",
              "4  84358402         M        20.29         14.34          135.10     1297.0   \n",
              "\n",
              "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
              "0          0.11840           0.27760          0.3001              0.14710   \n",
              "1          0.08474           0.07864          0.0869              0.07017   \n",
              "2          0.10960           0.15990          0.1974              0.12790   \n",
              "3          0.14250           0.28390          0.2414              0.10520   \n",
              "4          0.10030           0.13280          0.1980              0.10430   \n",
              "\n",
              "   ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \\\n",
              "0  ...          17.33           184.60      2019.0            0.1622   \n",
              "1  ...          23.41           158.80      1956.0            0.1238   \n",
              "2  ...          25.53           152.50      1709.0            0.1444   \n",
              "3  ...          26.50            98.87       567.7            0.2098   \n",
              "4  ...          16.67           152.20      1575.0            0.1374   \n",
              "\n",
              "   compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \\\n",
              "0             0.6656           0.7119                0.2654          0.4601   \n",
              "1             0.1866           0.2416                0.1860          0.2750   \n",
              "2             0.4245           0.4504                0.2430          0.3613   \n",
              "3             0.8663           0.6869                0.2575          0.6638   \n",
              "4             0.2050           0.4000                0.1625          0.2364   \n",
              "\n",
              "   fractal_dimension_worst  Unnamed: 32  \n",
              "0                  0.11890          NaN  \n",
              "1                  0.08902          NaN  \n",
              "2                  0.08758          NaN  \n",
              "3                  0.17300          NaN  \n",
              "4                  0.07678          NaN  \n",
              "\n",
              "[5 rows x 33 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-aeb09c2d-f3cc-4ce2-8ee6-67786aa2a175\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>842302</td>\n",
              "      <td>M</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>842517</td>\n",
              "      <td>M</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>84300903</td>\n",
              "      <td>M</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>84348301</td>\n",
              "      <td>M</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>84358402</td>\n",
              "      <td>M</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 33 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-aeb09c2d-f3cc-4ce2-8ee6-67786aa2a175')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-aeb09c2d-f3cc-4ce2-8ee6-67786aa2a175 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-aeb09c2d-f3cc-4ce2-8ee6-67786aa2a175');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 90
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes\n",
        "#dtypes for certain variables within the dataset"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kggROtGjGbfk",
        "outputId": "5a196322-e732-46e5-a406-1bb4d07a9d91"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "id                           int64\n",
              "diagnosis                  float64\n",
              "radius_mean                float64\n",
              "texture_mean               float64\n",
              "                            ...   \n",
              "concave points_worst       float64\n",
              "symmetry_worst             float64\n",
              "fractal_dimension_worst    float64\n",
              "Unnamed: 32                float64\n",
              "Length: 33, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df2 = pd.read_csv(\"Breast_Cancer.csv\") \n",
        "#This reads the data file which is named Breast_Cancer.csv\n",
        "\n",
        "df2.head()\n",
        "#df2[\"Size\"] = df2[\"Tumor Size\"]\n",
        "\n",
        "def transform(x):\n",
        "    x = x.replace(\" anaplastic; Grade IV\", \"4\")\n",
        "    return str(x)\n",
        "df2['Grade'] = df2['Grade'].apply(transform)\n",
        "df2['Grade'].value_counts()\n",
        "df2.head()\n"
      ],
      "metadata": {
        "id": "btooVokehT-x",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 392
        },
        "outputId": "de3ad0be-3cd6-4a86-c774-ee877a33bcc3"
      },
      "execution_count": 133,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Age   Race Marital Status T Stage  N Stage 6th Stage  \\\n",
              "0   68  White        Married       T1      N1       IIA   \n",
              "1   50  White        Married       T2      N2      IIIA   \n",
              "2   58  White       Divorced       T3      N3      IIIC   \n",
              "3   58  White        Married       T1      N1       IIA   \n",
              "4   47  White        Married       T2      N1       IIB   \n",
              "\n",
              "               differentiate Grade   A Stage  Tumor Size Estrogen Status  \\\n",
              "0      Poorly differentiated     3  Regional           4        Positive   \n",
              "1  Moderately differentiated     2  Regional          35        Positive   \n",
              "2  Moderately differentiated     2  Regional          63        Positive   \n",
              "3      Poorly differentiated     3  Regional          18        Positive   \n",
              "4      Poorly differentiated     3  Regional          41        Positive   \n",
              "\n",
              "  Progesterone Status  Regional Node Examined  Reginol Node Positive  \\\n",
              "0            Positive                      24                      1   \n",
              "1            Positive                      14                      5   \n",
              "2            Positive                      14                      7   \n",
              "3            Positive                       2                      1   \n",
              "4            Positive                       3                      1   \n",
              "\n",
              "   Survival Months Status  \n",
              "0               60  Alive  \n",
              "1               62  Alive  \n",
              "2               75  Alive  \n",
              "3               84  Alive  \n",
              "4               50  Alive  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-92a46160-c2c7-4430-8956-f965e445dab6\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Race</th>\n",
              "      <th>Marital Status</th>\n",
              "      <th>T Stage</th>\n",
              "      <th>N Stage</th>\n",
              "      <th>6th Stage</th>\n",
              "      <th>differentiate</th>\n",
              "      <th>Grade</th>\n",
              "      <th>A Stage</th>\n",
              "      <th>Tumor Size</th>\n",
              "      <th>Estrogen Status</th>\n",
              "      <th>Progesterone Status</th>\n",
              "      <th>Regional Node Examined</th>\n",
              "      <th>Reginol Node Positive</th>\n",
              "      <th>Survival Months</th>\n",
              "      <th>Status</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>68</td>\n",
              "      <td>White</td>\n",
              "      <td>Married</td>\n",
              "      <td>T1</td>\n",
              "      <td>N1</td>\n",
              "      <td>IIA</td>\n",
              "      <td>Poorly differentiated</td>\n",
              "      <td>3</td>\n",
              "      <td>Regional</td>\n",
              "      <td>4</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>24</td>\n",
              "      <td>1</td>\n",
              "      <td>60</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>50</td>\n",
              "      <td>White</td>\n",
              "      <td>Married</td>\n",
              "      <td>T2</td>\n",
              "      <td>N2</td>\n",
              "      <td>IIIA</td>\n",
              "      <td>Moderately differentiated</td>\n",
              "      <td>2</td>\n",
              "      <td>Regional</td>\n",
              "      <td>35</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>14</td>\n",
              "      <td>5</td>\n",
              "      <td>62</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>58</td>\n",
              "      <td>White</td>\n",
              "      <td>Divorced</td>\n",
              "      <td>T3</td>\n",
              "      <td>N3</td>\n",
              "      <td>IIIC</td>\n",
              "      <td>Moderately differentiated</td>\n",
              "      <td>2</td>\n",
              "      <td>Regional</td>\n",
              "      <td>63</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>14</td>\n",
              "      <td>7</td>\n",
              "      <td>75</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>58</td>\n",
              "      <td>White</td>\n",
              "      <td>Married</td>\n",
              "      <td>T1</td>\n",
              "      <td>N1</td>\n",
              "      <td>IIA</td>\n",
              "      <td>Poorly differentiated</td>\n",
              "      <td>3</td>\n",
              "      <td>Regional</td>\n",
              "      <td>18</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "      <td>84</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>47</td>\n",
              "      <td>White</td>\n",
              "      <td>Married</td>\n",
              "      <td>T2</td>\n",
              "      <td>N1</td>\n",
              "      <td>IIB</td>\n",
              "      <td>Poorly differentiated</td>\n",
              "      <td>3</td>\n",
              "      <td>Regional</td>\n",
              "      <td>41</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>3</td>\n",
              "      <td>1</td>\n",
              "      <td>50</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-92a46160-c2c7-4430-8956-f965e445dab6')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-92a46160-c2c7-4430-8956-f965e445dab6 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-92a46160-c2c7-4430-8956-f965e445dab6');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 133
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#df2[\"Status\"].value_counts()\n",
        "my_colors = list(islice(cycle(['g', 'r']), None, len(df)))\n",
        "\n",
        "\n",
        "df2.groupby('Grade')[\"Status\"].value_counts(normalize=True)\n",
        "df2.groupby('Grade')[\"Status\"].value_counts(normalize=True).plot(kind='bar', legend = True, color=my_colors)\n",
        "plt.legend(['Alive', 'Dead'],\n",
        "            loc='upper right', title='Status')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "zeQTrKlhFIyC",
        "outputId": "ecffd28c-81aa-44e3-f1a0-fedcbd9cbbf6"
      },
      "execution_count": 111,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x7feb8dfb6ca0>"
            ]
          },
          "metadata": {},
          "execution_count": 111
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "w_ZeK_TDSwKK"
      },
      "execution_count": 109,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "The histogram below shows how the number of tumors increase with age. The number of tumors spiked at age 45 in this graph."
      ],
      "metadata": {
        "id": "Lomy7uNUmqsn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df2.hist(column='Age')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 566
        },
        "id": "KCiLXlAtmeq8",
        "outputId": "ccf7d996-cca3-40dd-fbc3-e61b0acc721f"
      },
      "execution_count": 112,
      "outputs": [
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-112-cc916ff55ca8>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcolumn\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'Age'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mLegend\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_core.py\u001b[0m in \u001b[0;36mhist_frame\u001b[0;34m(data, column, by, grid, xlabelsize, xrot, ylabelsize, yrot, ax, sharex, sharey, figsize, layout, bins, backend, legend, **kwargs)\u001b[0m\n\u001b[1;32m    224\u001b[0m     \"\"\"\n\u001b[1;32m    225\u001b[0m     \u001b[0mplot_backend\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_get_plot_backend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbackend\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 226\u001b[0;31m     return plot_backend.hist_frame(\n\u001b[0m\u001b[1;32m    227\u001b[0m         \u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    228\u001b[0m         \u001b[0mcolumn\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcolumn\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_matplotlib/hist.py\u001b[0m in \u001b[0;36mhist_frame\u001b[0;34m(data, column, by, grid, xlabelsize, xrot, ylabelsize, yrot, ax, sharex, sharey, figsize, layout, bins, legend, **kwds)\u001b[0m\n\u001b[1;32m    462\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlegend\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mcan_set_label\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    463\u001b[0m             \u001b[0mkwds\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"label\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcol\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 464\u001b[0;31m         \u001b[0max\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcol\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdropna\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbins\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mbins\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    465\u001b[0m         \u001b[0max\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_title\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcol\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    466\u001b[0m         \u001b[0max\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgrid\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgrid\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/__init__.py\u001b[0m in \u001b[0;36minner\u001b[0;34m(ax, data, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1563\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0minner\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1564\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdata\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1565\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mmap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msanitize_sequence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1566\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1567\u001b[0m         \u001b[0mbound\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnew_sig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbind\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/axes/_axes.py\u001b[0m in \u001b[0;36mhist\u001b[0;34m(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, **kwargs)\u001b[0m\n\u001b[1;32m   6817\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mpatch\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   6818\u001b[0m                 \u001b[0mp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpatch\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 6819\u001b[0;31m                 \u001b[0mp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   6820\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mlbl\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   6821\u001b[0m                     \u001b[0mp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_label\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlbl\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py\u001b[0m in \u001b[0;36mupdate\u001b[0;34m(self, props)\u001b[0m\n\u001b[1;32m   1004\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1005\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mcbook\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_setattr_cm\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0meventson\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1006\u001b[0;31m             \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0m_update_property\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mprops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1007\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1008\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mret\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py\u001b[0m in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m   1004\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1005\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mcbook\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_setattr_cm\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0meventson\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1006\u001b[0;31m             \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0m_update_property\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mprops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1007\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1008\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mret\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py\u001b[0m in \u001b[0;36m_update_property\u001b[0;34m(self, k, v)\u001b[0m\n\u001b[1;32m    999\u001b[0m                 \u001b[0mfunc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'set_'\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1000\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mcallable\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfunc\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1001\u001b[0;31m                     raise AttributeError('{!r} object has no property {!r}'\n\u001b[0m\u001b[1;32m   1002\u001b[0m                                          .format(type(self).__name__, k))\n\u001b[1;32m   1003\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mv\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'Rectangle' object has no property 'legend'"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#df2.plot.scatter(x='Age', y='Tumor Size')\n",
        "#df2.plot.scatter(x='Grade', y='Tumor Size')\n",
        "#df2.hist(column='Age')\n",
        "#df2.plot.scatter(x='Tumor Size', y='Grade')\n",
        "df2.groupby(\"Grade\").Size.plot.hist(alpha=.5, density=False, legend=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 372
        },
        "id": "RySUyrmmH-Bs",
        "outputId": "b7bd9d75-fe9f-4bbd-e137-d9f6fcc9126c"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Grade\n",
              " anaplastic; Grade IV    AxesSubplot(0.125,0.125;0.775x0.755)\n",
              "1                        AxesSubplot(0.125,0.125;0.775x0.755)\n",
              "2                        AxesSubplot(0.125,0.125;0.775x0.755)\n",
              "3                        AxesSubplot(0.125,0.125;0.775x0.755)\n",
              "Name: Size, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 6
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df2mean = df2.mean()\n",
        "df2.groupby(by=[\"Grade\"]).mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 262
        },
        "id": "D6nnqJeScXNj",
        "outputId": "2b941e80-5c02-4a1a-9610-070ce51101c9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-14-6f7c54fb9e39>:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  df2mean = df2.mean()\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                             Age  Tumor Size  Regional Node Examined  \\\n",
              "Grade                                                                  \n",
              " anaplastic; Grade IV  52.315789   44.157895               14.473684   \n",
              "1                      55.289134   26.364641               12.675875   \n",
              "2                      54.322416   29.729051               14.387920   \n",
              "3                      52.615662   33.823582               15.111611   \n",
              "\n",
              "                       Reginol Node Positive  Survival Months  \n",
              "Grade                                                          \n",
              " anaplastic; Grade IV               6.157895        64.421053  \n",
              "1                                   3.068140        72.937385  \n",
              "2                                   3.922586        72.179073  \n",
              "3                                   5.154815        68.749775  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-405d9324-85c1-4cec-862f-856a05d2cca6\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Tumor Size</th>\n",
              "      <th>Regional Node Examined</th>\n",
              "      <th>Reginol Node Positive</th>\n",
              "      <th>Survival Months</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Grade</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>anaplastic; Grade IV</th>\n",
              "      <td>52.315789</td>\n",
              "      <td>44.157895</td>\n",
              "      <td>14.473684</td>\n",
              "      <td>6.157895</td>\n",
              "      <td>64.421053</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>55.289134</td>\n",
              "      <td>26.364641</td>\n",
              "      <td>12.675875</td>\n",
              "      <td>3.068140</td>\n",
              "      <td>72.937385</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>54.322416</td>\n",
              "      <td>29.729051</td>\n",
              "      <td>14.387920</td>\n",
              "      <td>3.922586</td>\n",
              "      <td>72.179073</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>52.615662</td>\n",
              "      <td>33.823582</td>\n",
              "      <td>15.111611</td>\n",
              "      <td>5.154815</td>\n",
              "      <td>68.749775</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-405d9324-85c1-4cec-862f-856a05d2cca6')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-405d9324-85c1-4cec-862f-856a05d2cca6 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-405d9324-85c1-4cec-862f-856a05d2cca6');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df2.plot.scatter(x='Tumor Size', y='Grade')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "bvFzu5o8czQk",
        "outputId": "fdf39cbc-152e-4d7f-9aed-844fc29af924"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f2348a13670>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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Fkgdd/l3d/9drv59CVoRkUNVV0Gr/6Md5tz9cYIjXSIiMgh0jlZERCRCCloREZEIKWhFREQipKAVERGJkIJWREQkQgpaERGRCCloRUREIqSgFRERiZCCVkREJEIKWhERkQgpaEVERCKkoBUREYmQglZERCRCCloREZEIKWhFREQipKAVERGJkIJWREQkQgpaERGRCCloRUREIqSgFRERiZCCVkREJEIKWhERkQgpaEVERCKkoBUREYmQglZERCRCCloREZEIKWhFREQipKAVERGJkIJWREQkQgpaERGRCCloRUREIqSgFRERiZCCVkREJEIKWhERkQgpaEVERCKkoBUREYmQglZERCRCCloREZEIKWhFREQipKAVERGJkIJWREQkQgpaERGRCCloRUREIqSgFRERiZCCVkREJEIKWhERkQgpaEVERCKkoBUREYmQglZERCRCCloREZEIKWhFREQipKAVERGJkIJWREQkQgpaERGRCCloRUREIqSgFRERiZCCVkREJEIFg12ARMvMioGVQBHBz/tOd/9qFOtq3NvClp1NNOxpZs2Wt5k/vYKaqeXUbmxk5foGNm7fTe3mt1kwawIXHv8eVm/exebGvTyxcSdzjhhFSVGC0kSM1xrfYXPjXl7cuocPHX0YxYk497+4jb3vtNLiXdcwoSxB/d7WnMuKC4zmA5kDjCiAUcXvPsaAOHAg67ETRxexYNYEfrvmLxnjl8ShMBHn7ea29rajJ47k/OqJ/P6lep7bvJsJIwsZVVpEUczZvKuFxn3vPv6S06fR1NLK6jd3s7B6Iqs37+Thl7ZzzMSRTCov49e1WzL6ps/hCe+t4KF12zj28FEUFxZw1b117N0PZYVw1XnV7XO7sHoiZcUFLF9bz6nTKzh20lj+8can2b7vAKOK4PgjD6MkDhsamzi/eiKfOW06QPvP7ZF1W3l5+z7eXzWKK889ltYDbWxqfIeCGLzW8A7zp1cAsHJ9Az96ZAMHwjlcXFPFvCPHMXX8yPbHzJk0hmmVIwH4yaPruaduKycfOY4ZE0czpXwEiYI4pYVx9u1vo2psCeVlRQCctnQFG99qZuq4Yh790hnc89xmlq+tZ8GxEzh/7qSMn1VqWWpbU+Ns2LaH1Zt3ta+namwJ37rvRR5+aTvzp5Xzqb+a3t439VxOryE1H6nndapPrno/e+uzPPJq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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df3 = pd.read_csv(\"BRCA.csv\") \n",
        "#This reads the data file which is named BRCA.csv\n",
        "\n",
        "df3.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 444
        },
        "id": "8n1IC2JVavm_",
        "outputId": "72bd1cf6-df6f-45c2-9c8f-0a7e055cc6cd"
      },
      "execution_count": 132,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     Patient_ID   Age  Gender  Protein1  Protein2  Protein3  Protein4  \\\n",
              "0  TCGA-D8-A1XD  36.0  FEMALE  0.080353   0.42638   0.54715  0.273680   \n",
              "1  TCGA-EW-A1OX  43.0  FEMALE -0.420320   0.57807   0.61447 -0.031505   \n",
              "2  TCGA-A8-A079  69.0  FEMALE  0.213980   1.31140  -0.32747 -0.234260   \n",
              "3  TCGA-D8-A1XR  56.0  FEMALE  0.345090  -0.21147  -0.19304  0.124270   \n",
              "4  TCGA-BH-A0BF  56.0  FEMALE  0.221550   1.90680   0.52045 -0.311990   \n",
              "\n",
              "  Tumour_Stage                      Histology ER status PR status HER2 status  \\\n",
              "0          III  Infiltrating Ductal Carcinoma  Positive  Positive    Negative   \n",
              "1           II             Mucinous Carcinoma  Positive  Positive    Negative   \n",
              "2          III  Infiltrating Ductal Carcinoma  Positive  Positive    Negative   \n",
              "3           II  Infiltrating Ductal Carcinoma  Positive  Positive    Negative   \n",
              "4           II  Infiltrating Ductal Carcinoma  Positive  Positive    Negative   \n",
              "\n",
              "                  Surgery_type Date_of_Surgery Date_of_Last_Visit  \\\n",
              "0  Modified Radical Mastectomy       15-Jan-17          19-Jun-17   \n",
              "1                   Lumpectomy       26-Apr-17          09-Nov-18   \n",
              "2                        Other       08-Sep-17          09-Jun-18   \n",
              "3  Modified Radical Mastectomy       25-Jan-17          12-Jul-17   \n",
              "4                        Other       06-May-17          27-Jun-19   \n",
              "\n",
              "  Patient_Status  \n",
              "0          Alive  \n",
              "1           Dead  \n",
              "2          Alive  \n",
              "3          Alive  \n",
              "4           Dead  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-97b4ae9a-7995-4b8f-b653-93e17b32c27a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Patient_ID</th>\n",
              "      <th>Age</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Protein1</th>\n",
              "      <th>Protein2</th>\n",
              "      <th>Protein3</th>\n",
              "      <th>Protein4</th>\n",
              "      <th>Tumour_Stage</th>\n",
              "      <th>Histology</th>\n",
              "      <th>ER status</th>\n",
              "      <th>PR status</th>\n",
              "      <th>HER2 status</th>\n",
              "      <th>Surgery_type</th>\n",
              "      <th>Date_of_Surgery</th>\n",
              "      <th>Date_of_Last_Visit</th>\n",
              "      <th>Patient_Status</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>TCGA-D8-A1XD</td>\n",
              "      <td>36.0</td>\n",
              "      <td>FEMALE</td>\n",
              "      <td>0.080353</td>\n",
              "      <td>0.42638</td>\n",
              "      <td>0.54715</td>\n",
              "      <td>0.273680</td>\n",
              "      <td>III</td>\n",
              "      <td>Infiltrating Ductal Carcinoma</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Negative</td>\n",
              "      <td>Modified Radical Mastectomy</td>\n",
              "      <td>15-Jan-17</td>\n",
              "      <td>19-Jun-17</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>TCGA-EW-A1OX</td>\n",
              "      <td>43.0</td>\n",
              "      <td>FEMALE</td>\n",
              "      <td>-0.420320</td>\n",
              "      <td>0.57807</td>\n",
              "      <td>0.61447</td>\n",
              "      <td>-0.031505</td>\n",
              "      <td>II</td>\n",
              "      <td>Mucinous Carcinoma</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Negative</td>\n",
              "      <td>Lumpectomy</td>\n",
              "      <td>26-Apr-17</td>\n",
              "      <td>09-Nov-18</td>\n",
              "      <td>Dead</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>TCGA-A8-A079</td>\n",
              "      <td>69.0</td>\n",
              "      <td>FEMALE</td>\n",
              "      <td>0.213980</td>\n",
              "      <td>1.31140</td>\n",
              "      <td>-0.32747</td>\n",
              "      <td>-0.234260</td>\n",
              "      <td>III</td>\n",
              "      <td>Infiltrating Ductal Carcinoma</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Negative</td>\n",
              "      <td>Other</td>\n",
              "      <td>08-Sep-17</td>\n",
              "      <td>09-Jun-18</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>TCGA-D8-A1XR</td>\n",
              "      <td>56.0</td>\n",
              "      <td>FEMALE</td>\n",
              "      <td>0.345090</td>\n",
              "      <td>-0.21147</td>\n",
              "      <td>-0.19304</td>\n",
              "      <td>0.124270</td>\n",
              "      <td>II</td>\n",
              "      <td>Infiltrating Ductal Carcinoma</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Negative</td>\n",
              "      <td>Modified Radical Mastectomy</td>\n",
              "      <td>25-Jan-17</td>\n",
              "      <td>12-Jul-17</td>\n",
              "      <td>Alive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>TCGA-BH-A0BF</td>\n",
              "      <td>56.0</td>\n",
              "      <td>FEMALE</td>\n",
              "      <td>0.221550</td>\n",
              "      <td>1.90680</td>\n",
              "      <td>0.52045</td>\n",
              "      <td>-0.311990</td>\n",
              "      <td>II</td>\n",
              "      <td>Infiltrating Ductal Carcinoma</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Positive</td>\n",
              "      <td>Negative</td>\n",
              "      <td>Other</td>\n",
              "      <td>06-May-17</td>\n",
              "      <td>27-Jun-19</td>\n",
              "      <td>Dead</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-97b4ae9a-7995-4b8f-b653-93e17b32c27a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-97b4ae9a-7995-4b8f-b653-93e17b32c27a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-97b4ae9a-7995-4b8f-b653-93e17b32c27a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 132
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df3.groupby('Tumour_Stage')[\"Patient_Status\"].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c-EH7pcTCTLv",
        "outputId": "ca14e4fb-8059-4563-933b-9af91b43a8f8"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Tumour_Stage  Patient_Status\n",
              "I             Alive              51\n",
              "              Dead               10\n",
              "II            Alive             144\n",
              "              Dead               38\n",
              "III           Alive              60\n",
              "              Dead               18\n",
              "Name: Patient_Status, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "QCsG0FI7XbGe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "tumor_c = list(islice(cycle(['b', 'y', 'r', 'g']), None, len(df)))\n",
        "df3.groupby('Tumour_Stage')[\"Surgery_type\"].value_counts(normalize=True).plot(kind='bar', legend=True, color = ['black', 'black', 'yellow', 'red', 'black', 'red', 'yellow','black','red','black','yellow','black'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 459
        },
        "id": "tE27aAGvWNow",
        "outputId": "ddac69fc-ee21-4d81-eec2-c66d1ed97ca4"
      },
      "execution_count": 130,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb8e0646d0>"
            ]
          },
          "metadata": {},
          "execution_count": 130
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAGoCAYAAABWnx4HAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOydd7gdVdX/P9+EhGAAQRIUpAQwgig9IBFeOogvSFFAY0QEFEVALK+CP5QmYMWGgKLSkapIkY4JgiAkgdCCQASVAEpAkCIt5Pv7Y89JTm7OLUlmz7ln7vo8z3numXJn7Zk9s86etVeRbYIgCILOZ1C7GxAEQRCUQyj0IAiCmhAKPQiCoCaEQg+CIKgJodCDIAhqQij0IAiCmtAnhS5pR0kPSpou6fAW2z8paaakqcXnU+U3NQiCIOiJxXrbQdJg4GRge2AGMEnS5banddn1QtsH91XwiBEjPGrUqAVpaxAEwYBnypQpT9se2Wpbrwod2ASYbvsRAEkXALsCXRX6AjFq1CgmT568KIcIgiAYcEj6e3fb+mJyeTvwWNPyjGJdVz4s6R5Jl0haeQHbGARBECwiZU2KXgGMsr0ucD1wVqudJB0gabKkyTNnzixJdBAEQQB9U+iPA80j7pWKdXOw/YztV4vFXwIbtTqQ7dNsj7E9ZuTIliagIAiCYCHpiw19EjBa0mokRf5R4GPNO0hawfaTxeIuwAML05jXX3+dGTNm8MorryzMvwcZGDZsGCuttBJDhgxpd1OCIOiFXhW67VmSDgauBQYDp9u+X9KxwGTblwOfl7QLMAv4N/DJhWnMjBkzWGqppRg1ahSSFuYQQYnY5plnnmHGjBmsttpq7W5OEAS90JcROravAq7qsu7Ipu9fA762qI155ZVXQpn3IySx3HLLEfMdQdAZ9LtI0VDm/YvojyDoHPqdQg+CIAgWjj6ZXNpF2aPDqM4UBEGdiRF6C44//nje/e53s+6667L++utz++23t7tJ3XLCCSe0uwlBnZAW7hP0C0Khd+G2227jyiuv5M477+See+7hhhtuYOWV+xb4OmvWrEWW/8YbbyzQ/qHQgyBoEAq9C08++SQjRoxg8cUXB2DEiBGsuOKKjBo1iqeffhqAyZMns9VWWwFw9NFHs/fee7PZZpux9957M3PmTLbffnve/e5386lPfYpVV111zv+de+65bLLJJqy//vp85jOfmaO8l1xySb785S+z3nrrcfzxx7PbbrvNac/111/P7rvv3rKthx9+OC+//DLrr78+48eP58gjj+RHP/rRnO1HHHEEP/7xj5k4cSJbbLEFO+20E2uuuSaf/exnmT17NgDXXXcdY8eOZcMNN2TPPffkxRdfLPeCBkFQHbbb8tloo43clWnTps2zDJT66QsvvPCC11tvPY8ePdoHHnigJ06caNteddVVPXPmTNv2pEmTvOWWW9q2jzrqKG+44Yb+73//a9s+6KCDfMIJJ9i2r776agOeOXOmp02b5p133tmvvfaabfvAAw/0WWedNec8L7zwQtv27Nmzveaaa/qpp56ybY8bN86XX355t+0dPnz4nO+PPvqoN9hgA9v2G2+84dVXX91PP/20J0yY4MUXX9x//etfPWvWLG+33Xa++OKLPXPmTP/P//yPX3zxRdv2t7/9bR9zzDG99ktQY2DhPkFlkOJ/WurVfj0p2g6WXHJJpkyZws0338yECRP4yEc+wre//e0e/2eXXXZhiSWWAOCWW27h0ksvBWDHHXdk2WWXBeDGG29kypQpbLzxxgC8/PLLLL/88gAMHjyYD3/4w0CaCN57770599xz2Xfffbnttts4++yz+9T2UaNGsdxyy3HXXXfxr3/9iw022IDlllsOgE022YTVV18dgHHjxnHLLbcwbNgwpk2bxmabbQbAa6+9xtixY/t8rYIg6F+EQm/B4MGD2Wqrrdhqq61YZ511OOuss1hsscXmmCm6piYYPnx4r8e0zT777MO3vvWt+bYNGzaMwYMHz1ned999+eAHP8iwYcPYc889WWyxvnfTpz71Kc4880z++c9/st9++81Z39VjSBK22X777Tn//PP7fPwgCPov/dqG3t1rxcJ++sKDDz7Iww8/PGd56tSprLrqqowaNYopU6YA8Jvf/Kbb/99ss8246KKLgGSffvbZZwHYdtttueSSS3jqqacA+Pe//83f/946rfGKK67IiiuuyHHHHce+++7bY3uHDBnC66+/Pmd5991355prrmHSpEm8//3vn7P+jjvu4NFHH2X27NlceOGFbL755my66ab86U9/Yvr06QC89NJLPPTQQz3KC4Kg/xIj9C68+OKLHHLIITz33HMstthivOMd7+C0007jgQceYP/99+cb3/jG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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "KpBr3dF_CTDm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df3.mean()\n",
        "#df3.Protein2.astype(int)\n",
        "df3.groupby(by=[\"Tumour_Stage\"]).mean()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 231
        },
        "id": "ckC49d6obYzp",
        "outputId": "1722fe4c-468d-4b61-8c2f-cad58f9bcbb9"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-9-1ad797f842ab>:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  df3.mean()\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                    Age  Protein1  Protein2  Protein3  Protein4\n",
              "Tumour_Stage                                                   \n",
              "I             62.359375 -0.014430  1.001318 -0.165147  0.037828\n",
              "II            59.052910 -0.007734  0.964763 -0.065409  0.018023\n",
              "III           55.753086 -0.094220  0.862207 -0.088845 -0.031453"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8288abba-e478-4103-95d1-e5ed88a715c4\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Protein1</th>\n",
              "      <th>Protein2</th>\n",
              "      <th>Protein3</th>\n",
              "      <th>Protein4</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Tumour_Stage</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>I</th>\n",
              "      <td>62.359375</td>\n",
              "      <td>-0.014430</td>\n",
              "      <td>1.001318</td>\n",
              "      <td>-0.165147</td>\n",
              "      <td>0.037828</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>II</th>\n",
              "      <td>59.052910</td>\n",
              "      <td>-0.007734</td>\n",
              "      <td>0.964763</td>\n",
              "      <td>-0.065409</td>\n",
              "      <td>0.018023</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>III</th>\n",
              "      <td>55.753086</td>\n",
              "      <td>-0.094220</td>\n",
              "      <td>0.862207</td>\n",
              "      <td>-0.088845</td>\n",
              "      <td>-0.031453</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8288abba-e478-4103-95d1-e5ed88a715c4')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-8288abba-e478-4103-95d1-e5ed88a715c4 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-8288abba-e478-4103-95d1-e5ed88a715c4');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df3.plot.scatter(x='Protein2', y='Tumour_Stage')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 570
        },
        "id": "59juQYlj7s17",
        "outputId": "cf01c229-a7d3-4fef-f0d3-3abaed4629b3"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-10-01c7e5f146a4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdf3\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'Protein2'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'Tumour_Stage'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_core.py\u001b[0m in \u001b[0;36mscatter\u001b[0;34m(self, x, y, s, c, **kwargs)\u001b[0m\n\u001b[1;32m   1634\u001b[0m             ...                       colormap='viridis')\n\u001b[1;32m   1635\u001b[0m         \"\"\"\n\u001b[0;32m-> 1636\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkind\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"scatter\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ms\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mc\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1637\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1638\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mhexbin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mC\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreduce_C_function\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgridsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_core.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    915\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mkind\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_dataframe_kinds\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    916\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mABCDataFrame\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 917\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mplot_backend\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkind\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkind\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    918\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    919\u001b[0m                 \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"plot kind {kind} can only be used for data frames\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_matplotlib/__init__.py\u001b[0m in \u001b[0;36mplot\u001b[0;34m(data, kind, **kwargs)\u001b[0m\n\u001b[1;32m     69\u001b[0m             \u001b[0mkwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"ax\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"left_ax\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0max\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     70\u001b[0m     \u001b[0mplot_obj\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mPLOT_CLASSES\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mkind\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 71\u001b[0;31m     \u001b[0mplot_obj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgenerate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     72\u001b[0m     \u001b[0mplot_obj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     73\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mplot_obj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_matplotlib/core.py\u001b[0m in \u001b[0;36mgenerate\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    286\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_compute_plot_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    287\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_setup_subplots\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 288\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_plot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    289\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_add_table\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    290\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_legend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/pandas/plotting/_matplotlib/core.py\u001b[0m in \u001b[0;36m_make_plot\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1068\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1069\u001b[0m             \u001b[0mlabel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1070\u001b[0;31m         scatter = ax.scatter(\n\u001b[0m\u001b[1;32m   1071\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1072\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/__init__.py\u001b[0m in \u001b[0;36minner\u001b[0;34m(ax, data, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1563\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0minner\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1564\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdata\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1565\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mmap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msanitize_sequence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1566\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1567\u001b[0m         \u001b[0mbound\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnew_sig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbind\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0max\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/cbook/deprecation.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    356\u001b[0m                 \u001b[0;34mf\"%(removal)s.  If any parameter follows {name!r}, they \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    357\u001b[0m                 f\"should be pass as keyword, not positionally.\")\n\u001b[0;32m--> 358\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    359\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    360\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/axes/_axes.py\u001b[0m in \u001b[0;36mscatter\u001b[0;34m(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, plotnonfinite, **kwargs)\u001b[0m\n\u001b[1;32m   4380\u001b[0m         \u001b[0;31m# Process **kwargs to handle aliases, conflicts with explicit kwargs:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4381\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4382\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_process_unit_info\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mxdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mydata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   4383\u001b[0m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconvert_xunits\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4384\u001b[0m         \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconvert_yunits\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/axes/_base.py\u001b[0m in \u001b[0;36m_process_unit_info\u001b[0;34m(self, xdata, ydata, kwargs)\u001b[0m\n\u001b[1;32m   2072\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2073\u001b[0m         \u001b[0mkwargs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_single_axis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mxdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxaxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'xunits'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2074\u001b[0;31m         \u001b[0mkwargs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_single_axis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mydata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0myaxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'yunits'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2075\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2076\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/axes/_base.py\u001b[0m in \u001b[0;36m_process_single_axis\u001b[0;34m(data, axis, unit_name, kwargs)\u001b[0m\n\u001b[1;32m   2054\u001b[0m                 \u001b[0;31m# We only need to update if there is nothing set yet.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2055\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhave_units\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2056\u001b[0;31m                     \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate_units\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2057\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2058\u001b[0m             \u001b[0;31m# Check for units in the kwargs, and if present update axis\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/axis.py\u001b[0m in \u001b[0;36mupdate_units\u001b[0;34m(self, data)\u001b[0m\n\u001b[1;32m   1514\u001b[0m         \u001b[0mneednew\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconverter\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0mconverter\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1515\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconverter\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconverter\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1516\u001b[0;31m         \u001b[0mdefault\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconverter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdefault_units\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1517\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdefault\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munits\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1518\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_units\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdefault\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/category.py\u001b[0m in \u001b[0;36mdefault_units\u001b[0;34m(data, axis)\u001b[0m\n\u001b[1;32m    105\u001b[0m         \u001b[0;31m# the conversion call stack is default_units -> axis_info -> convert\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    106\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munits\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 107\u001b[0;31m             \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_units\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mUnitData\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    108\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    109\u001b[0m             \u001b[0maxis\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munits\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/category.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, data)\u001b[0m\n\u001b[1;32m    173\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_counter\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mitertools\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcount\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    174\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdata\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 175\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    176\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    177\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mstaticmethod\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/category.py\u001b[0m in \u001b[0;36mupdate\u001b[0;34m(self, data)\u001b[0m\n\u001b[1;32m    210\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mval\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mOrderedDict\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfromkeys\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    211\u001b[0m             \u001b[0;31m# OrderedDict just iterates over unique values in data.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 212\u001b[0;31m             \u001b[0mcbook\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_isinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbytes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mval\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    213\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mconvertible\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    214\u001b[0m                 \u001b[0;31m# this will only be called so long as convertible is True.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.8/dist-packages/matplotlib/cbook/__init__.py\u001b[0m in \u001b[0;36m_check_isinstance\u001b[0;34m(_types, **kwargs)\u001b[0m\n\u001b[1;32m   2121\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mv\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2122\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtypes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2123\u001b[0;31m             raise TypeError(\n\u001b[0m\u001b[1;32m   2124\u001b[0m                 \"{!r} must be an instance of {}, not a {}\".format(\n\u001b[1;32m   2125\u001b[0m                     \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: 'value' must be an instance of str or bytes, not a float"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The new dataframe rd (below) was created to establish a relationship between \"radius_worst\" and \"radius_mean\" and its \"radius_se\" to understand how many standard errors away from the mean that *nearly all* of the diagnoses are *malignant*. This is a step we can use to help guide our question searching process."
      ],
      "metadata": {
        "id": "gM6HA-3tVMEu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "my_colors = list(islice(cycle(['y', 'g']), None, len(df)))\n",
        "\n",
        "\n",
        "rd = df[df[\"radius_worst\"] > (df[\"radius_mean\"] + 11 * df[\"radius_se\"])]\n",
        "rd[\"diagnosis\"].value_counts()\n",
        "rd[\"diagnosis\"].value_counts().plot(kind='bar', legend = True, color=my_colors)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "o8UNl7_vHXfa",
        "outputId": "7e187d44-1743-43ad-fd48-dc618e6caa07"
      },
      "execution_count": 102,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb8e0cc640>"
            ]
          },
          "metadata": {},
          "execution_count": 102
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD3CAYAAAAE2w/rAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAOqklEQVR4nO3df4zUdX7H8ddL1mPLAQuF8bTCuViVSBZUHPToGcDDM1RPqT8S/HFGKLJ/GKltToFrE84YDSRtpDZtavC0XFOz/EGVnljtmfMHNrHILrIVRHvGnt5STnex4K51owvv/sFg2BV3Zuf73V0/8Hwkhp3vfGe+78X4zNfPfGfGESEAQHpOGe4BAADVIeAAkCgCDgCJIuAAkCgCDgCJIuAAkKiaoTzYxIkTo76+figPCQDJa2lp6YiIQt/tQxrw+vp6NTc3D+UhASB5tt873naWUAAgUQQcABJFwAEgUUO6Bg7gxPP555+rra1N3d3dwz1K8mprazVp0iSdeuqpFe1PwAFk0tbWpjFjxqi+vl62h3ucZEWE9u/fr7a2Nk2ZMqWix7CEAiCT7u5uTZgwgXhnZFsTJkwY0P/JEHAAmRHvfAz075ElFAAnnPvuu0+jR4/Wxx9/rDlz5uiKK64YtllWr149aDMQ8IS99BJnPRh+dXXPqrPzky9ut7TMyvX5582r/ktn7r///hwn+frNwBIKgBPCgw8+qPPOO0+XXXaZ3n77bUnS4sWLtWnTJklHQjpr1iw1NDSosbFRR7+NbPv27ZoxY4YuvPBC3XvvvWpoaJAkbdiwQddff70WLFigc889VytWrPjiWE1NTZo+fboaGhq0cuVKSdKhQ4e0ePFiNTQ0aPr06Vq3bt2XZli1apWmTZumGTNm6J577sn8O3MGDiB5LS0t2rhxo3bu3Kmenh7NnDlTF198ca997rrrLq1evVqSdNttt2nLli265pprtGTJEj366KOaPXu2Vq1a1esxO3fu1Ouvv66RI0dq6tSpWr58uUaMGKGVK1eqpaVF48eP15VXXqnNmzdr8uTJ2rt3r3bt2iVJOnDgQK/n2r9/v5566im99dZbsv2l+6vBGTiA5L3yyiu67rrrNGrUKI0dO1bXXnvtl/Z58cUXdemll2r69Ol64YUXtHv3bh04cECdnZ2aPXu2JOmWW27p9Zj58+errq5OtbW1mjZtmt577z1t375d8+bNU6FQUE1NjW699VZt3bpVZ599tt59910tX75czz33nMaOHdvruY4+z9KlS/Xkk09q1KhRmX9vAg7ghNfd3a0777xTmzZt0htvvKFly5ZVdLneyJEjv/h5xIgR6unp+cp9x48fr9bWVs2bN0+PPPKI7rjjjl7319TU6LXXXtONN96oLVu2aMGCBdX/QiUEHEDy5syZo82bN+vTTz9VZ2ennn766V73H431xIkT1dXV9cWa9Lhx4zRmzBht27ZNkrRx48ayx7rkkkv08ssvq6OjQ4cOHVJTU5Pmzp2rjo4OHT58WDfccIMeeOAB7dixo9fjurq6dPDgQV111VVat26dWltbM//erIEDSN7MmTO1aNEiXXDBBTrttNM0a1bvK2HGjRunZcuWqaGhQaeffnqv+x977DEtW7ZMp5xyiubOnau6urp+j3XGGWdo7dq1uvzyyxURuvrqq7Vw4UK1trZqyZIlOnz4sCRpzZo1vR7X2dmphQsXqru7WxGhhx56KPPv7aOvxA6FYrEYfB54friMEF8HdXXP6pxzJvbaNmZMcZimGbiuri6NHj1akrR27Vrt27dPDz/88LDNs2fPHp1//vm9ttluiYgv/aVyBg7gpPbMM89ozZo16unp0VlnnaUNGzYM90gVI+AATmqLFi3SokWLhnuMqpR9EdP247Y/tL3rOPf9yHbYnni8xwIABk8lV6FskPSl611sT5Z0paT3c54JQFIOawhfSjuhDfQ1ybIBj4itkj46zl3rJK2QxL864CR26NA7Oniwh4hndPTzwGtrayt+TFVr4LYXStobEa18jCRwcvvkk/v0wQf3qaPjHB09J6yt3TO8QyXq6DfyVGrAAbc9StKf68jySSX7N0pqlKRvf/vbAz0cgK+5iP9VV9fdvbZddBGn40Ohmndi/r6kKZJabf9a0iRJO2yffrydI2J9RBQjolgoFKqfFADQy4DPwCPiDUmnHb1dingxIjpynAsAUEYllxE2SXpV0lTbbbaXDv5YAIByyp6BR8TNZe6vz20aAEDF+DRCAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEgUAQeARBFwAEhUJV9q/LjtD23vOmbb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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rd2 = df[df[\"radius_worst\"] > (df[\"radius_mean\"] + 9 * df[\"radius_se\"])]\n",
        "rd2[\"diagnosis\"].value_counts()\n",
        "rd2[\"diagnosis\"].value_counts().plot(kind='bar', legend = True, color=my_colors)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "CwKD6wRv8lFH",
        "outputId": "12137cf9-fb9e-45e8-bce6-745b523d6aa8"
      },
      "execution_count": 98,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb8e2505b0>"
            ]
          },
          "metadata": {},
          "execution_count": 98
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rd3 = df[df[\"radius_worst\"] > (df[\"radius_mean\"] + 7 * df[\"radius_se\"])]\n",
        "rd3[\"diagnosis\"].value_counts()\n",
        "rd3[\"diagnosis\"].value_counts().plot(kind='bar', legend = True, color=my_colors)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "uK79X9Py80an",
        "outputId": "74e862ef-d957-4285-fd67-a618750fc7a1"
      },
      "execution_count": 99,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb8e207040>"
            ]
          },
          "metadata": {},
          "execution_count": 99
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "rd4 = df[df[\"radius_worst\"] > (df[\"radius_mean\"] + 5 * df[\"radius_se\"])]\n",
        "rd4[\"diagnosis\"].value_counts()\n",
        "rd4[\"diagnosis\"].value_counts().plot(kind='bar', legend = True, color=my_colors)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "X8UCDS1X_LLN",
        "outputId": "761233e8-21b3-46e3-fde6-3ce255793c7c"
      },
      "execution_count": 100,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb8e180430>"
            ]
          },
          "metadata": {},
          "execution_count": 100
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.concavity_worst.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0NooeoTHkVc2",
        "outputId": "202cf465-51fe-463d-91f8-913484e4971a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.27218848330404216"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "This line of code finds the mean of the concave points of the cell nucleus of each person's tumor, wether malignant or benign."
      ],
      "metadata": {
        "id": "WJO-_ymdkdYH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Sets M equal to the dataframe's \"M\" column, which stands for malignant tumor patients\n",
        "M = df.set_index(\"diagnosis\").loc[\"M\"]\n",
        "display(M.head())\n",
        "M.concavity_worst.plot.hist()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 614
        },
        "id": "TmfK7-IXlMLB",
        "outputId": "0b60c388-1857-49cb-d2a4-00a30d793478"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "                 id  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "diagnosis                                                                   \n",
              "M            842302        17.99         10.38          122.80     1001.0   \n",
              "M            842517        20.57         17.77          132.90     1326.0   \n",
              "M          84300903        19.69         21.25          130.00     1203.0   \n",
              "M          84348301        11.42         20.38           77.58      386.1   \n",
              "M          84358402        20.29         14.34          135.10     1297.0   \n",
              "\n",
              "           smoothness_mean  compactness_mean  concavity_mean  \\\n",
              "diagnosis                                                      \n",
              "M                  0.11840           0.27760          0.3001   \n",
              "M                  0.08474           0.07864          0.0869   \n",
              "M                  0.10960           0.15990          0.1974   \n",
              "M                  0.14250           0.28390          0.2414   \n",
              "M                  0.10030           0.13280          0.1980   \n",
              "\n",
              "           concave points_mean  symmetry_mean  ...  texture_worst  \\\n",
              "diagnosis                                      ...                  \n",
              "M                      0.14710         0.2419  ...          17.33   \n",
              "M                      0.07017         0.1812  ...          23.41   \n",
              "M                      0.12790         0.2069  ...          25.53   \n",
              "M                      0.10520         0.2597  ...          26.50   \n",
              "M                      0.10430         0.1809  ...          16.67   \n",
              "\n",
              "           perimeter_worst  area_worst  smoothness_worst  compactness_worst  \\\n",
              "diagnosis                                                                     \n",
              "M                   184.60      2019.0            0.1622             0.6656   \n",
              "M                   158.80      1956.0            0.1238             0.1866   \n",
              "M                   152.50      1709.0            0.1444             0.4245   \n",
              "M                    98.87       567.7            0.2098             0.8663   \n",
              "M                   152.20      1575.0            0.1374             0.2050   \n",
              "\n",
              "           concavity_worst  concave points_worst  symmetry_worst  \\\n",
              "diagnosis                                                          \n",
              "M                   0.7119                0.2654          0.4601   \n",
              "M                   0.2416                0.1860          0.2750   \n",
              "M                   0.4504                0.2430          0.3613   \n",
              "M                   0.6869                0.2575          0.6638   \n",
              "M                   0.4000                0.1625          0.2364   \n",
              "\n",
              "           fractal_dimension_worst  Unnamed: 32  \n",
              "diagnosis                                        \n",
              "M                          0.11890          NaN  \n",
              "M                          0.08902          NaN  \n",
              "M                          0.08758          NaN  \n",
              "M                          0.17300          NaN  \n",
              "M                          0.07678          NaN  \n",
              "\n",
              "[5 rows x 32 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-1743de52-d97b-47ae-97bd-9a13ec4dae25\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>symmetry_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>diagnosis</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>M</th>\n",
              "      <td>842302</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>M</th>\n",
              "      <td>842517</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>M</th>\n",
              "      <td>84300903</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>M</th>\n",
              "      <td>84348301</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>M</th>\n",
              "      <td>84358402</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 32 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-1743de52-d97b-47ae-97bd-9a13ec4dae25')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-1743de52-d97b-47ae-97bd-9a13ec4dae25 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-1743de52-d97b-47ae-97bd-9a13ec4dae25');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7feb913c3ee0>"
            ]
          },
          "metadata": {},
          "execution_count": 26
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**A diagnosis of 'M' signifies that the cell is malignant. This graph is a histogram of all of the concavity points of malignant cells. This could help us later understand which concavity point most frequently signifies a cancerous cell. **"
      ],
      "metadata": {
        "id": "WbwOYn_zl9se"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 317
        },
        "id": "kqK9XHjtj2Ek",
        "outputId": "70f9eb75-21fe-4d42-f750-991329a1c262"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         id  diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "0    842302        1.0        17.99         10.38          122.80     1001.0   \n",
              "1    842517        1.0        20.57         17.77          132.90     1326.0   \n",
              "2  84300903        1.0        19.69         21.25          130.00     1203.0   \n",
              "3  84348301        1.0        11.42         20.38           77.58      386.1   \n",
              "4  84358402        1.0        20.29         14.34          135.10     1297.0   \n",
              "\n",
              "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
              "0          0.11840           0.27760          0.3001              0.14710   \n",
              "1          0.08474           0.07864          0.0869              0.07017   \n",
              "2          0.10960           0.15990          0.1974              0.12790   \n",
              "3          0.14250           0.28390          0.2414              0.10520   \n",
              "4          0.10030           0.13280          0.1980              0.10430   \n",
              "\n",
              "   ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \\\n",
              "0  ...          17.33           184.60      2019.0            0.1622   \n",
              "1  ...          23.41           158.80      1956.0            0.1238   \n",
              "2  ...          25.53           152.50      1709.0            0.1444   \n",
              "3  ...          26.50            98.87       567.7            0.2098   \n",
              "4  ...          16.67           152.20      1575.0            0.1374   \n",
              "\n",
              "   compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \\\n",
              "0             0.6656           0.7119                0.2654          0.4601   \n",
              "1             0.1866           0.2416                0.1860          0.2750   \n",
              "2             0.4245           0.4504                0.2430          0.3613   \n",
              "3             0.8663           0.6869                0.2575          0.6638   \n",
              "4             0.2050           0.4000                0.1625          0.2364   \n",
              "\n",
              "   fractal_dimension_worst  Unnamed: 32  \n",
              "0                  0.11890          NaN  \n",
              "1                  0.08902          NaN  \n",
              "2                  0.08758          NaN  \n",
              "3                  0.17300          NaN  \n",
              "4                  0.07678          NaN  \n",
              "\n",
              "[5 rows x 33 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-84f387c5-a55c-4358-b059-f07dfd4be506\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>842302</td>\n",
              "      <td>1.0</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>842517</td>\n",
              "      <td>1.0</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>84300903</td>\n",
              "      <td>1.0</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>84348301</td>\n",
              "      <td>1.0</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>84358402</td>\n",
              "      <td>1.0</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 33 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-84f387c5-a55c-4358-b059-f07dfd4be506')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-84f387c5-a55c-4358-b059-f07dfd4be506 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-84f387c5-a55c-4358-b059-f07dfd4be506');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df2.Age.mad())\n",
        "#how to do Tumor Size \n",
        "df2[\"Tumor Size\"].mad()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L7F_99Eqf4vo",
        "outputId": "a3e9432a-5b8d-48bb-d04d-cea07813cf18"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "7.5470121616227095\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "15.570934235541026"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**K-Nearest Neighbors Method**\n",
        "\n",
        "The code below uses the K-Nearest Neighbors method to predict whether the diagnosis of the tumor is malignant or benign. "
      ],
      "metadata": {
        "id": "2y1rM1ysmlQD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def transform(x):\n",
        "    x = x.replace(\"B\", \"0\").replace(\"M\", \"1\")\n",
        "    return float(x)\n",
        "df['diagnosis'] = df['diagnosis'].apply(transform)\n",
        "\n",
        "train = df.sample(frac=.5)\n",
        "val = df.drop(train.index)\n",
        "\n",
        "# Features in our model. All quantitative, except Neighborhood.\n",
        "features = [\"radius_mean\",\t\"texture_mean\",\t\"perimeter_mean\",\t\"area_mean\",\t\"smoothness_mean\",\t\"compactness_mean\",\t\"concavity_mean\"]\n",
        "\n",
        "X_train_dict = train[features].to_dict(orient=\"records\")\n",
        "X_val_dict = val[features].to_dict(orient=\"records\")\n",
        "\n",
        "y_train = train[\"diagnosis\"]\n",
        "y_val = val[\"diagnosis\"]\n",
        "\n",
        "from sklearn.feature_extraction import DictVectorizer\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsRegressor\n",
        "\n",
        "def get_val_error(X_train_dict, y_train, X_val_dict, y_val):\n",
        "    \n",
        "    # convert categorical variables to dummy variables\n",
        "    vec = DictVectorizer(sparse=False)\n",
        "    vec.fit(X_train_dict)\n",
        "    X_train = vec.transform(X_train_dict)\n",
        "    X_val = vec.transform(X_val_dict)\n",
        "\n",
        "    # standardize the data\n",
        "    scaler = StandardScaler()\n",
        "    scaler.fit(X_train)\n",
        "    X_train_sc = scaler.transform(X_train)\n",
        "    X_val_sc = scaler.transform(X_val)\n",
        "    \n",
        "    # Fit a 10-nearest neighbors model.\n",
        "    model = KNeighborsRegressor(n_neighbors=7)\n",
        "    model.fit(X_train_sc, y_train)\n",
        "    \n",
        "    # Make predictions on the validation set.\n",
        "    y_val_pred = model.predict(X_val_sc)\n",
        "    rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())\n",
        "    \n",
        "    return rmse"
      ],
      "metadata": {
        "id": "yifdCmk_YhHp"
      },
      "execution_count": 78,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "val_avg_1 = ((get_val_error(X_train_dict, y_train, X_val_dict, y_val)) + get_val_error(X_val_dict, y_val, X_train_dict, y_train))/2\n",
        "print(val_avg_1)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QmPNuY8Ila0V",
        "outputId": "0f095748-f9c6-4e60-b462-c71d4b92587b"
      },
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.2200316841057553\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = df[\"radius_mean\"]\n",
        "y = df['diagnosis']\n",
        "a = df[\"area_mean\"]\n",
        "r = np.corrcoef(x, y)\n",
        "r\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u6o6ZcAuIv6F",
        "outputId": "e003d7ba-2c39-4d41-c0a4-9793e862a7fb"
      },
      "execution_count": 89,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1.        , 0.73002851],\n",
              "       [0.73002851, 1.        ]])"
            ]
          },
          "metadata": {},
          "execution_count": 89
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Project Ideas**\n",
        "\n",
        "We have three main ideas of what model we are planning to build:\n",
        "\n",
        "\n",
        "1.   We plan to use mean radius and standard error of the radius of the tumor to predict whether the tumor is malignant or benign.\n",
        "2.   We plan to use the size of the tumor to predict the tumor stage (1, 2, 3, or 4).\n",
        "3. We plan to use the hormone protein levels to determine the type of surgery required to treat the cancer.\n",
        "\n"
      ],
      "metadata": {
        "id": "012V6hhTnale"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%%shell\n",
        "#pwd\n",
        "jupyter nbconvert --to html ///content/drive/MyDrive/DataScienceProject/Milestone2.ipynb"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V2CceUvGDFyb",
        "outputId": "be2041b5-355c-40a9-a3b1-c6ffc6d84b38"
      },
      "execution_count": 134,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[NbConvertApp] Converting notebook ///content/drive/MyDrive/DataScienceProject/Milestone2.ipynb to html\n",
            "[NbConvertApp] Writing 498967 bytes to ///content/drive/MyDrive/DataScienceProject/Milestone2.html\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 134
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#git clone https://github.com/bharatsolanky/bharatsolanky.github.io"
      ],
      "metadata": {
        "id": "6imBta2GFpJx"
      },
      "execution_count": 135,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "eWTaMI9PEZVv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "P4Qc5In__i4j"
      }
    }
  ]
}