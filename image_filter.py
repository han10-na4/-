import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import sys
import urllib.parse
import random

class ImageStylizer:
    def __init__(self):
        self.styles = {
            '1': '수채화',
            '2': '연필 스케치',
            '3': '오일 페인팅',
            '4': '만화 스타일',
            '5': '고대 사진',
            '6': '인상파',
            '7': '팝 아트',
            '8': '모자이크',
            '9': '흑백 사진',
            '10': '네온',
            '11': '스텐실',
            '12': '포스터',
            '13': '그라데이션',
            '14': '픽셀 아트',
            '15': '스플래터',
            '16': '아크릴화',
            '17': '파스텔화',
            '18': '잉크 드로잉',
            '19': '포토리얼리즘',
            '20': '하이퍼리얼리즘'
        }
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
    def normalize_path(self, path):
        """경로 정규화"""
        try:
            # 따옴표 제거 및 공백 제거
            path = path.strip().strip('"\'')
            
            # Windows 경로에서 백슬래시를 슬래시로 변환
            path = path.replace('\\', '/')
            
            # 경로 구분자 통일
            path = os.path.normpath(path)
            
            # 절대 경로로 변환
            abs_path = os.path.abspath(path)
            
            return abs_path
        except Exception as e:
            raise ValueError(f"경로 정규화 중 오류 발생: {str(e)}")
        
    def validate_image_path(self, path):
        """이미지 파일 경로 검증"""
        try:
            # 경로 정규화
            abs_path = self.normalize_path(path)
            
            # 파일 존재 여부 확인
            if not os.path.exists(abs_path):
                return False, f"파일이 존재하지 않습니다: {abs_path}"
            
            # 파일 확장자 확인
            _, ext = os.path.splitext(abs_path)
            if ext.lower() not in self.supported_formats:
                return False, f"지원하지 않는 파일 형식입니다: {ext}"
            
            return True, abs_path
        except Exception as e:
            return False, f"경로 검증 중 오류 발생: {str(e)}"
    
    def validate_output_path(self, filename):
        """출력 파일 경로 검증"""
        try:
            # 파일명에 확장자가 없으면 .jpg 추가
            if not os.path.splitext(filename)[1]:
                filename += '.jpg'
            return True, filename
        except Exception as e:
            return False, f"출력 경로 검증 중 오류 발생: {str(e)}"
            
    def get_current_directory(self):
        """현재 작업 디렉토리 반환"""
        return os.getcwd()
        
    def list_directory(self, path=None):
        """디렉토리 내용 나열"""
        try:
            if path is None:
                path = self.get_current_directory()
            
            # 디렉토리 존재 여부 확인
            if not os.path.exists(path):
                return False, f"디렉토리가 존재하지 않습니다: {path}"
            
            # 디렉토리 내용 읽기
            files = os.listdir(path)
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in self.supported_formats]
            
            return True, image_files
        except Exception as e:
            return False, f"디렉토리 내용을 읽을 수 없습니다: {str(e)}"
            
    def apply_watercolor(self, image):
        """수채화 효과"""
        try:
            # 이미지 크기 확인
            height, width = image.shape[:2]
            
            # 블러 효과
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            
            # 엣지 강화
            edges = cv2.Canny(image, 100, 200)
            edges = cv2.dilate(edges, None)
            
            # 색상 강화
            enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
            
            # 수채화 효과
            watercolor = cv2.stylization(enhanced, sigma_s=60, sigma_r=0.6)
            
            # 엣지와 색상 합성
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(watercolor, 0.8, edges, 0.2, 0)
            
            return result
        except Exception as e:
            raise ValueError(f"수채화 효과 적용 중 오류 발생: {str(e)}")
    
    def apply_pencil_sketch(self, image):
        """연필 스케치 효과"""
        try:
            # 이미지 크기 확인
            height, width = image.shape[:2]
            
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 가우시안 블러
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 엣지 검출
            edges = cv2.Canny(blurred, 30, 70)
            
            # 엣지 강화
            edges = cv2.dilate(edges, None)
            
            # 반전
            sketch = 255 - edges
            
            # 노이즈 추가
            noise = np.random.normal(0, 10, sketch.shape).astype(np.uint8)
            sketch = cv2.add(sketch, noise)
            
            # BGR로 변환
            result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            return result
        except Exception as e:
            raise ValueError(f"연필 스케치 효과 적용 중 오류 발생: {str(e)}")
    
    def apply_oil_painting(self, image):
        """오일 페인팅 효과"""
        try:
            # 이미지 크기 확인
            height, width = image.shape[:2]
            
            # 블러 효과로 브러시 스트로크 효과 생성
            blurred = cv2.GaussianBlur(image, (7, 7), 0)
            
            # 색상 강화
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = hsv[:,:,1] * 1.2  # 채도 증가
            hsv[:,:,2] = hsv[:,:,2] * 1.1  # 명도 증가
            oil = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 엣지 검출
            gray = cv2.cvtColor(oil, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지 마스크 생성 및 처리
            edges = cv2.dilate(edges, None)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # 엣지와 오일 페인팅 효과 결합
            result = cv2.addWeighted(oil, 0.9, edges, 0.1, 0)
            
            # 질감 추가
            noise = np.random.normal(0, 10, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)
            
            return result
        except Exception as e:
            raise ValueError(f"오일 페인팅 효과 적용 중 오류 발생: {str(e)}")
    
    def apply_cartoon(self, image):
        """만화 스타일 효과"""
        try:
            # 엣지 검출
            edges = cv2.Canny(image, 100, 200)
            # 색상 단순화
            color = cv2.bilateralFilter(image, 9, 300, 300)
            # 엣지와 색상 합성
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
        except Exception as e:
            raise ValueError(f"만화 스타일 효과 적용 중 오류 발생: {str(e)}")
    
    def apply_vintage(self, image):
        """고대 사진 효과"""
        try:
            # 세피아 톤
            sepia = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
            vintage = cv2.transform(image, sepia)
            # 노이즈 추가
            noise = np.random.normal(0, 10, vintage.shape).astype(np.uint8)
            vintage = cv2.add(vintage, noise)
            # 대비 감소
            vintage = cv2.convertScaleAbs(vintage, alpha=0.8, beta=0)
            return vintage
        except Exception as e:
            raise ValueError(f"고대 사진 효과 적용 중 오류 발생: {str(e)}")
    
    def apply_impressionist(self, image):
        """인상파 스타일 효과"""
        try:
            # 이미지 크기 축소
            small = cv2.resize(image, None, fx=0.5, fy=0.5)
            
            # 블러 효과
            blurred = cv2.GaussianBlur(small, (7, 7), 0)
            
            # 색상 강화
            enhanced = cv2.convertScaleAbs(blurred, alpha=1.3, beta=20)
            
            # 원래 크기로 복원
            result = cv2.resize(enhanced, (image.shape[1], image.shape[0]))
            
            return result
        except Exception as e:
            raise ValueError(f"인상파 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_pop_art(self, image):
        """팝 아트 스타일 효과"""
        try:
            # HSV로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 색상 강화
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
            
            # 명도 증가
            hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.2)
            
            # BGR로 변환
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 엣지 검출
            edges = cv2.Canny(image, 100, 200)
            edges = cv2.dilate(edges, None)
            
            # 엣지와 색상 합성
            result = cv2.bitwise_and(enhanced, enhanced, mask=edges)
            
            return result
        except Exception as e:
            raise ValueError(f"팝 아트 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_mosaic(self, image):
        """모자이크 스타일 효과"""
        try:
            # 이미지 크기 축소
            small = cv2.resize(image, None, fx=0.1, fy=0.1)
            
            # 다시 확대 (모자이크 효과)
            result = cv2.resize(small, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
            
            return result
        except Exception as e:
            raise ValueError(f"모자이크 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_black_white(self, image):
        """흑백 사진 스타일 효과"""
        try:
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 대비 조절
            result = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
            
            # BGR로 변환
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        except Exception as e:
            raise ValueError(f"흑백 사진 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_neon(self, image):
        """네온 스타일 효과"""
        try:
            # 엣지 검출
            edges = cv2.Canny(image, 100, 200)
            edges = cv2.dilate(edges, None)
            
            # HSV로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 채도와 명도 증가
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 2.0)
            hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.5)
            
            # BGR로 변환
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 엣지와 색상 합성
            result = cv2.bitwise_and(enhanced, enhanced, mask=edges)
            
            return result
        except Exception as e:
            raise ValueError(f"네온 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_stencil(self, image):
        """스텐실 아트 효과"""
        try:
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 엣지 검출
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지 강화
            edges = cv2.dilate(edges, None)
            
            # 반전
            stencil = 255 - edges
            
            # BGR로 변환
            result = cv2.cvtColor(stencil, cv2.COLOR_GRAY2BGR)
            
            return result
        except Exception as e:
            raise ValueError(f"스텐실 아트 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_poster(self, image):
        """포스터화 효과"""
        try:
            # HSV로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 색상 수 감소
            hsv[:,:,0] = np.floor(hsv[:,:,0] / 30) * 30
            hsv[:,:,1] = np.floor(hsv[:,:,1] / 50) * 50
            hsv[:,:,2] = np.floor(hsv[:,:,2] / 50) * 50
            
            # BGR로 변환
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return result
        except Exception as e:
            raise ValueError(f"포스터화 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_gradient(self, image):
        """그라데이션 아트 효과"""
        try:
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 그라데이션 마스크 생성
            height, width = gray.shape
            gradient = np.zeros((height, width), dtype=np.uint8)
            for i in range(width):
                gradient[:, i] = np.linspace(0, 255, height)
                
            # 그라데이션 적용
            result = cv2.bitwise_and(image, image, mask=gradient)
            
            return result
        except Exception as e:
            raise ValueError(f"그라데이션 아트 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_pixel_art(self, image):
        """픽셀 아트 효과"""
        try:
            # 이미지 크기 축소
            small = cv2.resize(image, None, fx=0.1, fy=0.1, 
                             interpolation=cv2.INTER_NEAREST)
            
            # 다시 확대 (픽셀 아트 효과)
            result = cv2.resize(small, (image.shape[1], image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
            
            return result
        except Exception as e:
            raise ValueError(f"픽셀 아트 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_splatter(self, image):
        """스플래터 아트 효과"""
        try:
            # 노이즈 생성
            noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
            
            # 노이즈 적용
            result = cv2.add(image, noise)
            
            # 블러 효과
            result = cv2.GaussianBlur(result, (5, 5), 0)
            
            return result
        except Exception as e:
            raise ValueError(f"스플래터 아트 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_oil_painting_enhanced(self, image):
        """유화 스타일 효과"""
        try:
            # 블러 효과
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # 색상 강화
            enhanced = cv2.convertScaleAbs(blurred, alpha=1.2, beta=10)
            
            # 엣지 검출
            edges = cv2.Canny(image, 100, 200)
            edges = cv2.dilate(edges, None)
            
            # 엣지와 색상 합성
            result = cv2.bitwise_and(enhanced, enhanced, mask=edges)
            
            # 질감 추가
            noise = np.random.normal(0, 20, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)
            
            return result
        except Exception as e:
            raise ValueError(f"유화 스타일 적용 중 오류 발생: {str(e)}")
            
    def apply_acrylic(self, image):
        """아크릴화 효과"""
        try:
            # 색상 강화
            enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
            
            # 대비 증가
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 매트 효과
            result = cv2.addWeighted(enhanced, 0.8, 
                                   np.zeros_like(enhanced), 0.2, 0)
            
            return result
        except Exception as e:
            raise ValueError(f"아크릴화 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_pastel(self, image):
        """파스텔화 효과"""
        try:
            # 블러 효과
            blurred = cv2.GaussianBlur(image, (7, 7), 0)
            
            # 색상 부드럽게
            enhanced = cv2.convertScaleAbs(blurred, alpha=0.8, beta=30)
            
            # 크레용 효과
            noise = np.random.normal(0, 10, enhanced.shape).astype(np.uint8)
            result = cv2.add(enhanced, noise)
            
            return result
        except Exception as e:
            raise ValueError(f"파스텔화 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_ink_drawing(self, image):
        """잉크 드로잉 효과"""
        try:
            # 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 엣지 검출
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지 강화
            edges = cv2.dilate(edges, None)
            
            # 반전
            ink = 255 - edges
            
            # BGR로 변환
            result = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)
            
            return result
        except Exception as e:
            raise ValueError(f"잉크 드로잉 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_photorealism(self, image):
        """포토리얼리즘 효과"""
        try:
            # 색상 보정
            enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
            
            # 선명도 증가
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(enhanced, -1, kernel)
            
            return result
        except Exception as e:
            raise ValueError(f"포토리얼리즘 효과 적용 중 오류 발생: {str(e)}")
            
    def apply_hyperrealism(self, image):
        """하이퍼리얼리즘 효과"""
        try:
            # 색상 보정
            enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
            
            # 선명도 증가
            kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
            result = cv2.filter2D(enhanced, -1, kernel)
            
            # 미세한 질감 추가
            noise = np.random.normal(0, 5, result.shape).astype(np.uint8)
            result = cv2.add(result, noise)
            
            return result
        except Exception as e:
            raise ValueError(f"하이퍼리얼리즘 효과 적용 중 오류 발생: {str(e)}")
            
    def process_image(self, image_path, style_name):
        """이미지 처리"""
        try:
            # 경로 정규화
            image_path = self.normalize_path(image_path)
            
            # 이미지 읽기
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")
                
            # 이미지 읽기 시도 (한글 경로 지원)
            try:
                # np.fromfile과 cv2.imdecode를 사용하여 한글 경로 처리
                image_array = np.fromfile(image_path, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
            except Exception as e:
                raise ValueError(f"이미지 읽기 실패: {str(e)}")

            # 이미지 크기 확인 및 조정
            height, width = image.shape[:2]
            if height > 4000 or width > 4000:
                scale = min(4000/height, 4000/width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image = cv2.resize(image, (new_width, new_height))
                print(f"이미지 크기가 너무 커서 {new_width}x{new_height}로 조정되었습니다.")

            # 스타일 적용
            style_methods = {
                '수채화': self.apply_watercolor,
                '연필 스케치': self.apply_pencil_sketch,
                '오일 페인팅': self.apply_oil_painting,
                '만화 스타일': self.apply_cartoon,
                '고대 사진': self.apply_vintage,
                '인상파': self.apply_impressionist,
                '팝 아트': self.apply_pop_art,
                '모자이크': self.apply_mosaic,
                '흑백 사진': self.apply_black_white,
                '네온': self.apply_neon,
                '스텐실': self.apply_stencil,
                '포스터': self.apply_poster,
                '그라데이션': self.apply_gradient,
                '픽셀 아트': self.apply_pixel_art,
                '스플래터': self.apply_splatter,
                '아크릴화': self.apply_acrylic,
                '파스텔화': self.apply_pastel,
                '잉크 드로잉': self.apply_ink_drawing,
                '포토리얼리즘': self.apply_photorealism,
                '하이퍼리얼리즘': self.apply_hyperrealism
            }

            if style_name not in style_methods:
                raise ValueError(f"지원하지 않는 스타일입니다: {style_name}")

            result = style_methods[style_name](image)
            return result
        except Exception as e:
            raise ValueError(f"이미지 처리 중 오류 발생: {str(e)}")

    def save_image(self, image, output_path):
        """이미지 저장"""
        try:
            # 고정된 저장 디렉토리 설정
            save_dir = r"C:\과제\사진저장"
            
            # 디렉토리 생성
            os.makedirs(save_dir, exist_ok=True)
            
            # 파일명 추출
            filename = os.path.basename(output_path)
            
            # 파일 확장자 확인 및 추가
            _, ext = os.path.splitext(filename)
            if not ext:
                filename = filename + '.jpg'  # 기본 확장자 추가
                ext = '.jpg'
            
            # 전체 경로 생성
            full_path = os.path.join(save_dir, filename)
            
            # 이미지 저장 (한글 경로 지원)
            if ext.lower() in ['.jpg', '.jpeg']:
                # JPEG 저장
                ret, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ret:
                    with open(full_path, 'wb') as f:
                        f.write(buf.tobytes())
                else:
                    raise ValueError("JPEG 이미지 인코딩에 실패했습니다.")
            elif ext.lower() == '.png':
                # PNG 저장
                ret, buf = cv2.imencode(ext, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if ret:
                    with open(full_path, 'wb') as f:
                        f.write(buf.tobytes())
                else:
                    raise ValueError("PNG 이미지 인코딩에 실패했습니다.")
            else:
                # 기타 형식
                ret, buf = cv2.imencode(ext, image)
                if ret:
                    with open(full_path, 'wb') as f:
                        f.write(buf.tobytes())
                else:
                    raise ValueError("이미지 인코딩에 실패했습니다.")
                
            # 저장된 파일 확인
            if not os.path.exists(full_path):
                raise ValueError("이미지가 저장되었지만 파일을 찾을 수 없습니다.")
                
            # 파일 크기 확인
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                raise ValueError("저장된 파일의 크기가 0입니다.")
                
            print(f"이미지가 성공적으로 저장되었습니다: {full_path}")
            print(f"저장된 파일 크기: {file_size} bytes")
        except Exception as e:
            raise ValueError(f"이미지 저장 중 오류 발생: {str(e)}")

    def main(self):
        """메인 함수"""
        print("\n=== 이미지 스타일 변환기 ===")
        print("1. 스타일 선택하여 변환")
        print("2. 랜덤 스타일로 변환")
        print("3. 종료")
        
        while True:
            try:
                choice = input("\n메뉴를 선택하세요 (1-3): ").strip()
                
                if choice == '3':
                    print("프로그램을 종료합니다.")
                    break
                    
                if choice not in ['1', '2']:
                    print("잘못된 선택입니다. 1, 2, 3 중에서 선택해주세요.")
                    continue
                
                # 이미지 경로 입력
                while True:
                    image_path = input("\n변환할 이미지의 경로를 입력하세요: ").strip()
                    if not image_path:
                        print("경로를 입력해주세요.")
                        continue
                    try:
                        # 경로 정규화 및 검증
                        image_path = self.normalize_path(image_path)
                        if not os.path.exists(image_path):
                            print(f"파일이 존재하지 않습니다: {image_path}")
                            print(f"현재 디렉토리: {os.getcwd()}")
                            continue
                        break
                    except Exception as e:
                        print(f"경로 처리 중 오류 발생: {str(e)}")
                        continue
                
                # 스타일 선택
                if choice == '1':
                    print("\n사용 가능한 스타일:")
                    # 스타일 목록을 2열로 나누어 출력
                    style_items = list(self.styles.items())
                    half = len(style_items) // 2
                    for i in range(half):
                        left = style_items[i]
                        right = style_items[i + half] if i + half < len(style_items) else None
                        if right:
                            print(f"{left[0]:>2}. {left[1]:<15} // {right[0]:>2}. {right[1]:<15}")
                        else:
                            print(f"{left[0]:>2}. {left[1]:<15}")
                    
                    style_key = input("\n스타일을 선택하세요 (1-20): ").strip()
                    if style_key not in self.styles:
                        print("잘못된 스타일 선택입니다.")
                        continue
                    style_name = self.styles[style_key]
                else:
                    style_key = random.choice(list(self.styles.keys()))
                    style_name = self.styles[style_key]
                    print(f"\n선택된 랜덤 스타일: {style_name}")
                
                # 이미지 처리
                try:
                    result = self.process_image(image_path, style_name)
                except Exception as e:
                    print(f"\n오류가 발생했습니다: {str(e)}")
                    print("다시 시도해주세요.")
                    continue
                
                # 저장 경로 입력
                while True:
                    filename = input("\n저장할 파일명을 입력하세요 (확장자 없이): ").strip()
                    if not filename:
                        print("파일명을 입력해주세요.")
                        continue
                    try:
                        # 경로 정규화 및 검증
                        success, save_path = self.validate_output_path(filename)
                        if not success:
                            print(save_path)  # 오류 메시지 출력
                            continue
                        break
                    except Exception as e:
                        print(f"경로 처리 중 오류 발생: {str(e)}")
                        continue
                
                # 이미지 저장
                try:
                    self.save_image(result, save_path)
                    print(f"\n이미지가 성공적으로 저장되었습니다: {save_path}")
                except Exception as e:
                    print(f"\n이미지 저장 중 오류 발생: {str(e)}")
                    print("다시 시도해주세요.")
                    continue
                
            except Exception as e:
                print(f"\n오류가 발생했습니다: {str(e)}")
                print("다시 시도해주세요.")
                continue

if __name__ == "__main__":
    try:
        stylizer = ImageStylizer()
        stylizer.main()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n예기치 않은 오류가 발생했습니다: {str(e)}")
        sys.exit(1) 