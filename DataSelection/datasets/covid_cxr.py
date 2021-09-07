#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import PIL

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import SimpleITK as sitk

test_split = [
    "1e0331390a7b04e39b13de366476f6844d1d2828ab81e3a513a5c282ee2c8e55",
    "e7fc23033792866fd7e8210ba900f266cb8beecb1071919e5139510b23cb6846",
    "60d7a2e8b9445c3be4da99fd1c8e7661a9819747a24b8250fa0b3ae3b3720fd2",
    "46b5743b717e4e47805a010716fd1d86f38a8e7b6d9cdeb0209fc72271abef18",
    "23770b06626fff471f4f062e8faa8f58809eac82d243d0754ff3f7c78b7ac5b4",
    "50a57e2b29c31340a2730771a9376c0bae311a0fb93c430297b831ce60abe83f",
    "3d45de6f94dd60491b21722dcc24a3a7bbbf3e21e7ba2cf82fe828b5ff06eedb",
    "5dde2706d0aa7f4defb39fcc6e632c66f7af721021d88bd12d9e32d76b5ba2da",
    "5f793a2ab6158aab3e3366c8bece621892637a603ade8974381cbc9f29732b7d",
    "022b7be761919b30f189240b5e1ec1ca9960a1dc860de31f678f70df9bbddc45",
    "c931804097314c65abff6b63935efa8e87e23f8772e375eb5c82236fb29c109b",
    "96a65ff7e9cd0cca5b355ba1ee60cbcba2b2d9d0a25a31b6460e8a9b532ba69e",
    "d78b595239490f8bdd444c93b55001fcb643b82ddcea86c57c3e8e4be741f86c",
    "a93f669fc47832ead940590e03e231f626cb4fa799b2391eff7c9a5990fe0f69",
    "29f40c1ec6001a8e16c6643667b5f07047e0df93787fd44a91617cd53de86463",
    "d5f61e24c77ce617c0a406372c0325688dec30a09e35c05fde0296f0447ffbd3",
    "61f2117cb7cff78b04984870be9c3ab39395eb986b2d9ba3d3fd7d6fbd5ba38c",
    "ed62ac8bba5706bf5b4e361a00440c718af37d9374480ed439108a2137296409",
    "cdf454855068ad4fe60de915340b2d4b264fea9a7e408d92d90a687922c018a7",
    "20088a9294e3ab3cd2cb9a2ba6389acbe4d5551923ee63f8ceb49a17e02dedd8",
    "8ea40d73277789a0f966a81b897a869693dc0d4df39a371e2ef82fb38bfa8825",
    "6c95b3ce0ab40f14a386264766ccee1e684146c04e655868c4542b239bff44ef",
    "7656349a849c076d0c51de2ff2e11bc106e3016fec5f9e11fef281a38529e135",
    "d9ae2e58b74c15275af1476e22467e8a64920c4c679680932e5e24d3b32bd6d8",
    "db0840c8b1cb32d019db1d06855698a0c2704ccac9d82644d5690bd13f55c4a1",
    "2b9026f442831827886fc7d274f2173380c2821a4c72dd69a40f16e49ffd876d",
    "9c1f55ecb0baabd8ff44ed818d3fa867afcef602e5077552fc3ca489bde26ebf",
    "06125610311e2f4457f50a0fa931c0d50bc9c963a4aad0c862f3e2f932919fd1",
    "2c615937da9ecc1e87c8474388162fabb8b9ff42ce12693310f4444194c20dae",
    "9f000de752c7e955d3a9a4e8d070a02a9ab776b5c212470ad440f0fc0aac0aaf",
    "acf5f3ff65ccd17ce37dda4dae6a2798d737b68defc309beaa8330f8305bfd3d",
    "379e08804f302dfd7a0ce237eadf3f87a76ea785cb0d29b264dcb827608212b0",
    "b9f4bcb8ff429611bd9ed7e420f0dada5c2f75241e01183d4b67a4414628d2c1",
    "45008cf71524a1ed294804ca1b7da3da99cb6828e3d0008ed20418967847856b",
    "be8a4935911a4dbd99a2c893df0a95277013a5c9e8fc32e5e98017a9f020bb3d",
    "221c79ff6244c20854b7c1b7678c6dd27d97b33766f2c11670b090fc68c29488",
    "04cebb2a7f607275352d807d2dc5ea46f522ebef3762bd430153fab98bdc03ed",
    "91d3a3cc1fb80fb90cd34e0e63c80dd773facba6d077492fb7d175e8c3b9bb8f",
    "8ac8d1b9824d2f7250b05765ab155c619cf5b12cc1b6a8cb12c7a1006bdf4299",
    "72d608411891e1ae2a7ddcf48ff068a937763f251ed7994963558e735f95d51a",
    "01f68a3a4a6d90597bcaeb00cc767b2b44c6bc39be34c73b87cc7949efdb7edf",
    "df7be41bb9734ab8a4b2fa0554866ddabb7bf7adb2ff32f26d8e66d4749705f6",
    "62bee1c358aa4bc5dc4309999a00b5c31c7f0a6749ee4361057f7f3c01c17404",
    "fffa6cc2bcdddae231eb6977ca1f58132212a1aa8d84a65aeb1327d287b97948",
    "9331bea878f6be6265c1d5850c57a2064f7ba55443a362c43a2a93f27e867982",
    "b36ee39331347db05f7dfc064cabf6a7784e18cd0bd4b1f1dfa613becdee91c1",
    "d094b3b064e2a797542da83c308829f649479467756daddca9003184c8a99d92",
    "d9df8e97e2a4b0de7753d1a6fd97f50ca5ff373c13668fcb6b1e762c919e1296",
    "3a6fafbb624c9ca73ccdad01a03ca35fab9b202fa2129b4af8fecedaef9b58ea",
    "28d4f96d80833edf69f9988e46ab822f9bbcee6de0b3795a5d586a027f605e4e",
    "c4804d9d1485078608f1ec1548bc7e39cf567291890437f1415fddb36a3ac30a",
    "bceba9028fae6e92ca238f200b1b972d3239e19cb36af10b375a116e2e98ae3b",
    "57b13709d7732602ca5a2ec5c3df0ee914acfdd94a7fc67f0e86c12464e2cb13",
    "6bff488a42588d2e65f4f8d9ae5cfa76a8e673080ae30db8ac64a3f05a5e35ad",
    "75a7eb81145c6fb051f53a71547266de2c095b70ca9f4d859e44c7dfe3ddb1f4",
    "4deb8aeee6d0f185d27feada9219ad52784a122a301e3712a34d9fd7f6add5ed",
    "3324c89a2ee2e10d9d9e57aff2ccdcd50feafbdabab9d3d837a7473f36303dce",
    "2da1f4eb8be2a32b2c3a8c6e0e3f53591959414e6bad1d546620cbea4abbd23f",
    "52a32976bb404020c99bfa48980e3f3bfbcd4a0263389512e065b80aaa576b7a",
    "611643a7b159950f6ce668212ca71d5edf37c3b565e59cedeb08f58954dc4a38",
    "2557bee5f7ffe38ecbb4c18417f6f5979b2710cfff10f31039193f522bebb568",
    "6eda904af228434dc8513390297d22840c2ced845d7e2242676758be9810236f",
    "0e898e66f1a5ed24567bad984ca57da449ae2c7373cd263eaee51492a86fe8c2",
    "193b3fc0c10c128f9102f4715255ea940efc07bde5328b4b98b01fad3143b07b",
    "8e77b7dee8c1513a9cd757111733159197e4ddd96e351b5f3d845280ffe64b7f",
    "84dd4ed3de2aac524ab6c56f3b2e28452caf19d1dc3b807cb8f394201e6e3231",
    "e7046dd5a470c59edbe78700da9909e3211a065672193dbb1e7d525ada002711",
    "347b3ece514e44b1db3ba31f7a9b8597c15c89c700b983c0ab83d27e1720f04d",
    "e20d5fd6cb36ebd6ef36e64c44a58d8b541253b94589fcb38ed508a3f68a915f",
    "c0148578f6ad6595b181e48888a48f63665c4594cf67ed7f0b4713465f44bcd8",
    "de4dcff3765afa702e41ea8d18a0f9a345acca229d725b3bae1a6ce718d7452a",
    "e23b5e9633542fa68d1bbf39dbab1589d8dec92537ba401047b8ddabf95a988f",
    "5aaabaa9ca274157dfc50447bba49b1a1899e76bc665dbb66421f8077409245b",
    "ff548e2008fbdc71ea892922a0a2729cb01392efe21505670226ad5d9838bd23",
    "c6df4efaad2345c2c6d7698dbf1510ffa723f2c1bf55bb3adfbe67f9d2d3187e",
    "ab941e9b34477c1971ce6d067fa2f6b9df39e4dd3210c5194528e1caeaca2a80",
    "a51f1da153331847fc0a86c9825537f18510411e618035910afc6e7571912692",
    "49ed4d29fdb085a9aabe3c8573cc5881a3753db98fa243fb8e48a7556f835426",
    "5e2f7c13702861dedb7db30290d6365692f5fd777eddd7e4995dbac0896d5ce7",
    "280ff200fb3473c77c1650ea63a1f24354fdbb1e24c9b46c3c010b2a79d89465",
    "b5420d6a1c39035e585ac2a6fe187cba0b156f4dfc602c19cac2c4a8212e9ee1",
    "dfd5cf227d4c7c0f9c6c9447d5e41b760350d373cad0e18ecbf31dc7323701b6",
    "306a97309ebbd2d3ce146e8fe078752023619beb3884952eca08616269b9b9ad",
    "fc19c34690bcf65104199d0fdd252fc38dd15cc1b6c777c079915be435c3c43a",
    "376fed12de9a3937fc174c34e83e021c8d7406712ebbcb39576060ed485dba4b",
    "377127acb73161f54a7165685614f1f4a4e4eff6e481678c8850454abe2a8a05",
    "2cff968a7ced66339c79f80f20ae5556c417ee30829656e01d9631978e5d260c",
    "6637426422ab3558aa35804b23685aa2a7e346ee8fc9a09e852a5206cdaf1767",
    "2fc5c236dd5dcaff19107d53571c8e548d98ccbd7fb0ca3a0c09b8a3d1572b65",
    "f149acec5c699156f6fbfd6be98acd72412b50ea16340469b500d3214a016233",
    "eee4f1fcd1baa4168eeac830dd317a2cab424c174bad778a3744c0091993ee20",
    "c6be8f46f8bd560cddb42244b7d3c5e29e0a017e7a8cac85ea3de31bba941982",
    "0da34c1812d0650f26e006ff1d45f8c3df1ecb6456331a0247f061ee82e8bf38",
    "79b4b18831c033ce66e1d52f5598069085002bc60d74f52c20714cc5b32a39f3",
    "0efea0ef2b97736e932520be1ff087e9cb1e5a192bbbea7eba8c0bf97dfd70ff",
    "29ce7863f4fed579b808ebac55f0e6acce7f31fdbdd7d72deae76da1414bec6c",
    "8914271d2b6416816dcb8327c9f87e246e217e048ee56cf54197d0f2becfa33d",
    "92056ecc6d0d57f9e50800de876d18a54da721ee9e269fba9427a4e9da765eb3",
    "5a5ed190d9431b42c0cd5c6ad79f3afbe0d62d9d7354ac79c04bb86cf9a2dc49",
    "368ad7796c84318cdcd320bc8a5328a7d879aa201fb0b7f875a7b678c02042c9",
    "35364226777ca7527355d7d6509e62d8ac60d6a9d0b66a6ee57f5c8472fd7c40",
    "a981db19f4b2ecc8165cbedb3ea5e2b77a3cc13568e7efbe2ba87846535a5e09",
    "3815814a062ae14dd7d8372cfe3d5634b7bba9ada0e8694e1830905857c84bd6",
    "f4011cad1e443baf57ede6a25ceb0a2163bc4b49b8b2b64de13c9b1dc4f79791",
    "d54e415553717f10151eab13244fa53fee845faeddb33f0f0c78bd70f2c8a52b",
    "caf0a5c4d99aa3a05a51b5dc860d0e284a49c35843f12ea4b5554b509b8dfeea",
    "77607e19cfd88e6e3710792ca6aeee8cb90094361cbb529a7fc7f70516663057",
    "7499097de10f4f8d193b4ce80782fd60296cb87d2d789a61a09e7b71540d9dd5",
    "e35e3efcf6794a73b7261819bafe3588c3490639aa39e77db1df3c620e47f650",
    "9213ba2dbf2a836d9e7b2c78ff6da9bbe8323333aeba216247816a8361ace177",
    "39883003a61e4b94a228c60a494c3cce7e4d04e1548a3a51b78d651609c11794",
    "ee16b6e7eebfc7f9d8625451425759c737e7ca75de18b36ee08b89a3b9dac40e",
    "ba1c87380edf894fcc775a1bc17ac2eeb3d81c946a1ca27809048ca8ba23005f",
    "7454c3981e1cadfe51a77207c421e81b9923c5a87e2122355502d4246d445349",
    "9516db4a2c14680e9235541b9341a19ec70529dd56f878fa7610467179368aa0",
    "5e830c64c0966d28a8a91b719ae92dfef70a0a4ee171b4cd21ff664976235c8c",
    "c53ad67bd1ee8b047194c06714261c52e54b4005be0b8d4a3d702e09e4eed716",
    "4e1e43ddac0849d2f84512979b93697d620e228a6be1facec2eb29b69c62e95e",
    "ec394cc664db9ba0551fc9914bb0c4daab4a579a22bc90f71b596fe5445a5c04",
    "f6a4699a2683f16043df595872cb943cb6b4bb0412a0cddca226d1ddcf97720f"]


class CovidCXR(Dataset):
    def __init__(self, data_directory: str,
                 use_training_split: bool,
                 train_fraction: float = 0.8,
                 seed: int = 1234,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 num_samples: int = None,
                 csv_to_ignore: Optional[Path] = None) -> None:
        """
        Class for the UHB Dataset.

        :param data_directory: the directory containing all training images.
        :param use_training_split: whether to return the training or the validation split of the dataset.
        :param train_fraction: the proportion of samples to use for training
        :param seed: random seed to use for dataset creation
        :param shuffle: whether to shuffle the dataset prior to spliting between validation and training
        :param transform:
        :param num_samples: number of the samples to return (has to been smaller than the dataset split)
        param: csv_to_ignore: csv containing the list of series id to exclude from the dataset.
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            logging.error(
                f"The data directory {self.data_directory} does not exist. Make sure to upload the data first.")

        self.train = use_training_split
        self.train_fraction = train_fraction
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset_dataframe = pd.read_csv(self.data_directory / "dataset.csv")
        if csv_to_ignore is not None:
            df_to_ignore = pd.read_csv(self.data_directory / csv_to_ignore)
            series_to_ignore = np.unique(df_to_ignore.series.values)
            logging.info(f"Excluding {len(series_to_ignore)} subjects")
            self.dataset_dataframe = self.dataset_dataframe[~self.dataset_dataframe.series.isin(series_to_ignore)]
        self.transforms = transform

        # ------------- Clean original dataset ------------- #
        is_val_id = self.dataset_dataframe.series.apply(lambda x: x in test_split).values
        orig_labels = self.dataset_dataframe.label.values.astype(np.int64)
        scan_ids = self.dataset_dataframe.series.values
        folder_ids = self.dataset_dataframe.subject.values
        self.num_classes = 2
        self.num_datapoints = scan_ids.shape[0]
        # ------------- Split the data into training and validation sets ------------- #
        all_indices = np.arange(len(self.dataset_dataframe))
        train_indices = all_indices[~is_val_id]
        val_indices = all_indices[is_val_id]
        train_indices = self.random_state.permutation(train_indices) \
            if shuffle else train_indices
        val_indices = self.random_state.permutation(val_indices) if shuffle else val_indices
        self.indices = train_indices if use_training_split else val_indices
        if num_samples is not None:
            assert 0 < num_samples <= len(self.indices)
            self.indices = self.indices[:num_samples]
        self.num_samples = self.indices.shape[0]
        self.ids = scan_ids[self.indices]
        self.orig_labels = orig_labels[self.indices].reshape(-1)
        self.folder_ids = folder_ids[self.indices]
        self.targets = self.orig_labels

        # Identify case ids for ambiguous and clear label noise cases
        self.ambiguity_metric_args: Dict = dict()

        dataset_type = "TRAIN" if use_training_split else "VAL"
        print(self.targets.shape[0])
        print(f"Proportion of positive labels - {dataset_type}: {np.mean(self.targets)}")

    def __getitem__(self, index: int) -> Tuple[int, PIL.Image.Image, int]:
        """

        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        scan_id = self.ids[index]
        folder_id = self.folder_ids[index]
        filename = self.data_directory / folder_id / scan_id / "CR.dcm"
        target = self.targets[index]
        scan_image = self.load_dicom_image(filename)[0]
        max, min = scan_image.max(), scan_image.min()
        scan_image = (scan_image - min) / max
        scan_image = Image.fromarray(scan_image)
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        if scan_image.shape == 2:
            scan_image = scan_image.unsqueeze(dim=0)
        return index, scan_image, int(target)

    def __len__(self) -> int:
        """
        :return: The size of the dataset
        """
        return len(self.indices)

    def get_label_names(self) -> List[str]:
        return ["Normal", "Covid"]

    def get_selected_df(self, indices: np.ndarray) -> pd.DataFrame:
        """
        Get the subset of the dataframe given by an array of indices.
        :param indices: array of indices
        """
        df = pd.DataFrame(columns=["series", "dataset_index"])
        df["series"] = self.ids[indices]
        df["dataset_index"] = indices
        df = pd.merge(df, self.dataset_dataframe, on="series")
        return df

    @staticmethod
    def load_dicom_image(path: Path) -> np.ndarray:
        """
        Loads an array from a single dicom file.
        :param path: The path to the dicom file.
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        image = reader.Execute()
        pixels = sitk.GetArrayFromImage(image)
        # Return a float array, we may resize this in load_3d_images_and_stack, and interpolation will not work on int
        return pixels.astype(np.float)
