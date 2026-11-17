import streamlit as st
import numpy as np
from PIL import Image
from scipy import signal
import scipy.ndimage as nd
import matplotlib.pyplot as plt

# 1. 웹페이지 제목 및 설명
st.set_page_config(page_title="Object Counting App", layout="wide")
st.title("🔍 Object Counting Web Application (Homework #5)")
st.write("이미지를 업로드하면 노이즈 제거 및 이진화(Thresholding)를 거쳐 객체의 개수와 크기를 분석합니다.")

# 2. 사이드바: 파라미터 조절 기능
st.sidebar.header("⚙️ 파라미터 설정")
kernel_size = st.sidebar.slider("Median Filter Kernel Size (홀수)", min_value=1, max_value=15, value=7, step=2)
th_val = st.sidebar.slider("Threshold 값 (0~255)", min_value=0, max_value=255, value=150, step=1)

# 3. 이미지 파일 업로드 받기
uploaded_file = st.file_uploader("분석할 이미지 파일을 선택하세요 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지를 Grayscale로 열기
    I = Image.open(uploaded_file).convert('L')
    f = np.array(I)

    # 노이즈 제거 (Median Filter)
    f_denoised = signal.medfilt2d(f, kernel_size=kernel_size)

    # 이진화 (Thresholding)
    f_mask = f_denoised > th_val

    # 연결 요소 라벨링 (Connected Component Labeling)
    labels, nums = nd.label(f_mask)

    st.divider()
    
    # 4. 중간 처리 과정 시각화 (4단계)
    st.subheader("1. 영상 처리 단계별 결과")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(f, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Grayscale Image')
        fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(f_denoised, cmap='gray')
        ax2.axis('off')
        ax2.set_title(f'Denoised (k={kernel_size})')
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(f_mask, cmap='gray')
        ax3.axis('off')
        ax3.set_title(f'Binary Mask (Th={th_val})')
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        ax4.imshow(labels, cmap='Spectral_r')
        ax4.axis('off')
        ax4.set_title(f'Labeled (Total: {nums})')
        st.pyplot(fig4)

    # 히스토그램 출력
    with st.expander("📊 픽셀 밝기 히스토그램 (Intensity Histogram) 보기"):
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.hist(f_denoised.ravel(), bins=256, range=(0, 256), color='gray')
        ax_hist.axvline(th_val, color='red', linestyle='--', label=f'Threshold ({th_val})')
        ax_hist.set_xlabel('Pixel Intensity')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()
        ax_hist.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_hist)

    st.divider()

    # 5. Object Counting 및 최대/최소 객체 분석 결과
    st.subheader("2. Object Counting 및 크기 분석 결과")

    if nums > 0:
        # 각 라벨별 픽셀 수 계산 (기존 코드 로직 유지)
        cc = np.zeros((nums, 2))
        for i in range(1, nums + 1):
            cc[i - 1, 0] = i
            cc[i - 1, 1] = np.sum(labels == i)

        ind_max = np.argmax(cc[:, 1])
        max_label = int(cc[ind_max, 0])
        max_pixels = int(cc[ind_max, 1])

        ind_min = np.argmin(cc[:, 1])
        min_label = int(cc[ind_min, 0])
        min_pixels = int(cc[ind_min, 1])

        # 핵심 수치 요약 카드 출력
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("총 검출된 객체 수 (Total Objects)", f"{nums} 개")
        m_col2.metric(f"가장 큰 객체 (Label #{max_label})", f"{max_pixels:,} pixels")
        m_col3.metric(f"가장 작은 객체 (Label #{min_label})", f"{min_pixels:,} pixels")

        # 최종 3분할 비교 시각화
        fig_res, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        axes[0].imshow(f, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Input Image')

        axes[1].imshow(labels == max_label, cmap='Spectral_r')
        axes[1].axis('off')
        axes[1].set_title(f'Largest Component (Label {max_label}: {max_pixels} px)')

        axes[2].imshow(labels == min_label, cmap='Spectral_r')
        axes[2].axis('off')
        axes[2].set_title(f'Smallest Component (Label {min_label}: {min_pixels} px)')

        plt.tight_layout()
        st.pyplot(fig_res)

        # 세부 데이터 표로 보기
        with st.expander("📋 전체 객체별 픽셀 수 데이터 보기"):
            st.dataframe(
                {"Component Index (Label)": cc[:, 0].astype(int), "Number of Pixels": cc[:, 1].astype(int)},
                use_container_width=True
            )
    else:
        st.warning("현재 Threshold 값에서는 검출된 객체가 없습니다. 왼쪽 사이드바에서 Threshold 값을 낮춰보세요.")