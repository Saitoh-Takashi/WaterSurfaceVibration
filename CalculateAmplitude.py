import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from dataclasses import dataclass


@dataclass
class Range:
    """時間範囲を保持するクラス"""
    start: float  # 開始
    stop: float  # 終了


@dataclass
class ImageArea:
    """始点(x1, y1)と終点(x2, yx)の値および入力された画像をトリミングするメソッドを保持するクラス"""
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, img: np.ndarray) -> np.ndarray:
        """入力された画像をトリミングした画像を返す

        Args:
            img (np.ndarray): 入力画像

        Returns:
            np.ndarray: トリミング画像

        """
        return img[self.y1:self.y2 + 1, self.x1: self.x2 + 1]


def water_surface_height(gray_img: np.ndarray, roi: ImageArea, index: int) -> np.float64:
    """水面の位置 [pixel]をエッジ検出で算出．
    検出された点群のy座標 [pixel]の平均値を返す．

    Args:
        gray_img (np.ndarray): グレースケール画像
        roi (ImageArea): ROI
        index (int): フレーム番号

    Returns:
        np.float64: 水面の位置 [pixel]

    """
    img = roi.clip(gray_img)
    canny = cv2.Canny(img, threshold1=50, threshold2=50)
    cv2.imwrite(f'pic/{index:0=5}.jpg', canny)
    surface_points = np.where(canny != 0)

    return np.average(surface_points[0])


def container_displacement(gray_img: np.ndarray, roi: ImageArea, template: np.ndarray) -> np.float64:
    """容器の変位をテンプレートマッチングで算出

    Args:
        gray_img (np.ndarrray): グレースケール画像
        roi (ImageArea): ROI
        template (ImageArea): テンプレート画像

    Returns:
        np.float64: マッチング位置のy座標 [pixel]

    """
    img = roi.clip(gray_img)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return np.float64(max_loc[1])


def vibration_dataframe(file_path: str,
                        canny_roi: ImageArea,
                        tracking_roi: ImageArea,
                        image_resolution: float,
                        time_range: Range) -> pd.DataFrame:
    """水面の振動波形などを記録したデータフレームを返す．
    テンプレートマッチングで算出した容器の変位を用いて，エッジ検出で算出した振動波形から容器の変位成分を除去し，
    単位を[pixel] -> [m]に変換，データフレームに記録する．．

    Args:
        file_path (str): 動画のファイルパス
        canny_roi (ImageArea): エッジ検出用のROI
        tracking_roi (ImageArea): テンプレートマッチング用のROI
        image_resolution (float): 画像分解能 [m/pixel]
        time_range (Range): ピークの時間範囲

    Returns:
        pd.DataFrame: 計算結果のデータフレーム

    """
    # 動画，テンプレートの読み込み
    cap = cv2.VideoCapture(file_path)
    template = cv2.cvtColor(cv2.imread('template.png'), cv2.COLOR_BGR2GRAY)

    print(cap.get(cv2.CAP_PROP_FPS))

    # 結果用データフレームの初期化
    result_df = pd.DataFrame(index=pd.Series(
        np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 1 / cap.get(cv2.CAP_PROP_FPS))),
        columns=['ContainerDisplacement [pixel]', 'WaterSurfaceDisplacement [pixel]', 'Displacement [m]', 'Peak'])
    result_df.index.name = 'Time [s]'

    # 前フレームで振動波形を計算
    frame_index = 0
    while True:
        # フレーム読み込み
        ret, img = cap.read()

        if frame_index == 0:
            cv2.imwrite('test.jpg', img)

        if ret:
            # グレースケール変換
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 容器の変位を算出
            result_df['ContainerDisplacement [pixel]'].iloc[frame_index] = container_displacement(gray_img=gray_img,
                                                                                                  roi=tracking_roi,
                                                                                                  template=template)
            # 水面の振動波形を算出
            result_df['WaterSurfaceDisplacement [pixel]'].iloc[frame_index] = water_surface_height(gray_img=gray_img,
                                                                                                   roi=canny_roi,
                                                                                                   index=frame_index)
            frame_index += 1

        else:
            break

    # 0点オフセット
    result_df['ContainerDisplacement [pixel]'] = (result_df['ContainerDisplacement [pixel]'] -
                                                  result_df['ContainerDisplacement [pixel]'][0])
    result_df['WaterSurfaceDisplacement [pixel]'] = (result_df['WaterSurfaceDisplacement [pixel]'] -
                                                     result_df['WaterSurfaceDisplacement [pixel]'][
                                                         (time_range.start <= result_df.index) &
                                                         (result_df.index <= time_range.stop)].mean())

    # 単位変換 [pixel] -> [m]
    result_df['WaterSurfaceDisplacement [m]'] = (result_df['WaterSurfaceDisplacement [pixel]'] -
                                                 result_df['ContainerDisplacement [pixel]']) * image_resolution

    # ピークの算出
    for index in argrelmax(result_df['WaterSurfaceDisplacement [m]'].fillna(0).values, order=2):
        result_df['Peak'].iloc[index] = result_df['WaterSurfaceDisplacement [m]'].values[index]

    return result_df


def natural_frequency(peak_series: pd.Series, time_range: Range) -> np.float64:
    """振動波形のピークの間隔から固有振動数 [Hz]を算出

    Args:
        peak_series (pd.Series): ピークのSeries
        time_range (Range): ピークの時間範囲

    Returns:
        np.float64: 固有振動数 [Hz]

    """
    series = peak_series.dropna()
    series = pd.Series(series.index[(time_range.start <= series.index) & (series.index <= time_range.stop)])
    return 1 / series.diff().dropna().mean()


def damping_ratio(peak_series: pd.Series, time_range: Range) -> np.float64:
    """振動波形のピーク比から減衰比 [-]を算出

    Args:
        peak_series (pd.Series): ピークのSeries
        time_range (Range): ピークの時間範囲

    Returns:
        np.float64: 減衰比 [-]

    """
    series = peak_series.dropna()
    series = series[(time_range.start <= series.index) & (series.index <= time_range.stop)]
    return 2 * np.pi * ((1 / (len(series) - 1)) * np.log(series.iloc[0] / series.iloc[-1]))


def plot(result_df: pd.DataFrame, omega_n: np.float64, zeta: np.float64, m: float, time_range: Range) -> None:
    """結果をグラフに表示する.

    Args:
        result_df (pd.DataFrame): 結果のデータフレーム
        omega_n (np.float64): 固有振動数 [Hz]
        zeta (np.float64): 減衰比 [-]
        m (float): 水の重さ [kg]
        time_range (Range): ピークの時間範囲

    """
    k = m * (omega_n ** 2)
    c = 2 * m * zeta * omega_n

    plt.figure(figsize=(8, 6))
    plt.title(f'ωn = {omega_n:.3f} [Hz], ζ = {zeta:.3f} [-]\nm = {m:.3f} [kg], k = {k:.3f} [N/m], c = {c:.3f} [N s/m]')
    plt.plot(result_df['WaterSurfaceDisplacement [m]'])
    plt.plot(result_df['Peak'][(time_range.start <= result_df.index) & (result_df.index <= time_range.stop)], 'ro')
    plt.xlabel('Time [s]')
    plt.ylabel('Water Surface Displacement [m]')
    plt.savefig('result.svg')
    plt.show()

    plt.plot(result_df['WaterSurfaceDisplacement [pixel]'])
    plt.savefig('surface.svg')

    plt.plot(result_df['ContainerDisplacement [pixel]'])
    plt.savefig('container.svg')


def main():
    # 設定
    file_path = 'C:/video/60fps/MOV_0608.mp4'
    image_resolution = 0.015 / 243  # [m/pixel]
    canny_roi = ImageArea(x1=449, y1=692, x2=620, y2=1341)
    tracking_roi = ImageArea(x1=373, y1=140, x2=674, y2=454)
    time_range = Range(start=5.5, stop=5.5 + 1.3404)

    # 各種変位のデータ
    result_df = vibration_dataframe(file_path=file_path,
                                    canny_roi=canny_roi,
                                    tracking_roi=tracking_roi,
                                    image_resolution=image_resolution,
                                    time_range=time_range)

    # 振動パラメータの計算
    omega_n = natural_frequency(peak_series=result_df['Peak'], time_range=time_range)
    zeta = damping_ratio(peak_series=result_df['Peak'], time_range=time_range)

    # プロット
    plot(result_df=result_df, omega_n=omega_n, zeta=zeta, m=0.26, time_range=time_range)

    # データ保存
    result_df.to_csv('Result.csv', index=True)


if __name__ == '__main__':
    main()
