# # Image Filter 프로그램 분석

## 1. 클래스 구조
### `ImageStylizer` 클래스
```python
class ImageStylizer:
    def __init__(self):
        self.styles = {...}  # 20가지 스타일 정의
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
```

## 2. 주요 기능

### 2.1 경로 관리 기능
- `normalize_path()`: 파일 경로 정규화
- `validate_image_path()`: 이미지 파일 경로 검증
- `validate_output_path()`: 출력 파일 경로 검증
- `get_current_directory()`: 현재 작업 디렉토리 반환
- `list_directory()`: 디렉토리 내용 나열

### 2.2 스타일 변환 기능 (20가지)
#### 기본 스타일
1. **예술적 스타일**
   ```python
   def apply_watercolor(self, image)  # 수채화
   def apply_oil_painting(self, image)  # 유화
   def apply_acrylic(self, image)  # 아크릴화
   def apply_pastel(self, image)  # 파스텔화
   ```

2. **드로잉 스타일**
   ```python
   def apply_pencil_sketch(self, image)  # 연필 스케치
   def apply_ink_drawing(self, image)  # 잉크 드로잉
   def apply_cartoon(self, image)  # 만화 스타일
   ```

3. **특수 효과**
   ```python
   def apply_neon(self, image)  # 네온
   def apply_mosaic(self, image)  # 모자이크
   def apply_pixel_art(self, image)  # 픽셀 아트
   def apply_gradient(self, image)  # 그라데이션
   ```

4. **사진 스타일**
   ```python
   def apply_vintage(self, image)  # 고대 사진
   def apply_black_white(self, image)  # 흑백 사진
   def apply_photorealism(self, image)  # 포토리얼리즘
   def apply_hyperrealism(self, image)  # 하이퍼리얼리즘
   ```

## 3. 이미지 처리 파이프라인

### 3.1 이미지 전처리
```python
# 이미지 크기 조정
if height > 4000 or width > 4000:
    scale = min(4000/height, 4000/width)
    image = cv2.resize(image, (new_width, new_height))
```

### 3.2 이미지 저장 기능
```python
def save_image(self, image, output_path):
    save_dir = r"C:\과제\사진저장"
    # JPEG, PNG 등 다양한 포맷 지원
```

## 4. 주요 알고리즘 특징

### 4.1 색상 처리
- HSV 색공간 변환
- 채도/명도 조정
- CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용

### 4.2 필터 효과
- 가우시안 블러
- 엣지 검출 (Canny)
- 노이즈 생성/제거

## 5. 에러 처리
```python
try:
    # 작업 수행
except Exception as e:
    raise ValueError(f"에러 메시지: {str(e)}")
```

## 6. 사용자 인터페이스
### 6.1 메인 메뉴
1. 스타일 선택하여 변환
2. 랜덤 스타일로 변환
3. 종료

### 6.2 스타일 선택 UI
- 2열 형식으로 스타일 목록 표시
- 직관적인 번호 선택 방식

## 7. 성능 최적화
- 대용량 이미지 자동 리사이징
- 한글 경로 지원
- 메모리 효율적 처리

## 8. 확장성
- 새로운 스타일 추가 용이
- 다양한 이미지 포맷 지원
- 모듈화된 코드 구조

이 프로그램은 OpenCV를 기반으로 한 이미지 처리 라이브러리로, 사용자가 쉽게 다양한 예술적 효과를 이미지에 적용할 수 있도록 설계되었습니다.
