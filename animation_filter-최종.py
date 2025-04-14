import cv2
import numpy as np
import os
import sys
import urllib.parse
import random
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

class AnimationStylizer:
    def __init__(self):
        # CUDA 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 장치: {self.device}")
        
        # 모델 초기화 (최적화된 설정)
        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "safety_checker": None,  # 안전 필터 비활성화
            "low_cpu_mem_usage": True
        }
        
        # 모델 초기화
        self.ghibli_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",  # 기본 모델로 변경
            **kwargs
        ).to(self.device)
        
        self.disney_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",  # 기본 모델로 변경
            **kwargs
        ).to(self.device)
        
        # 얼굴 감지 모델 초기화
        self.face_detector = None
        try:
            # Haar Cascade 모델 파일 경로
            cascade_path = "haarcascade_frontalface_default.xml"
            
            # 모델 파일이 없으면 다운로드
            if not os.path.exists(cascade_path):
                print("얼굴 감지 모델 파일을 다운로드합니다...")
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, cascade_path)
            
            # OpenCV 얼굴 감지기 초기화
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            if self.face_detector.empty():
                raise ValueError("얼굴 감지 모델을 로드할 수 없습니다.")
            print("얼굴 감지기 초기화 완료")
        except Exception as e:
            print(f"얼굴 감지 모델 초기화 중 오류 발생: {str(e)}")
            print("얼굴 감지 기능이 제한될 수 있습니다.")
        
        # 스타일 정의
        self.styles = {
            '1': '디즈니 스타일',
            '2': '픽사 스타일',
            '3': '지브리 스타일'
        }
        
        # 디즈니/픽사 스타일 프롬프트
        self.disney_prompts = [
            "modern disney pixar style, young asian character, (anatomically correct facial features:1.3), natural messy black hair with volume, black t-shirt, turquoise and orange beaded necklace, black wireless earbud, (expressive human-like eyes:1.2), natural warm smile showing teeth, soft realistic skin shading, (detailed hair strands:1.3), subtle facial expressions, purple studio lighting",
            "disney animation style, asian youth with realistic proportions, (human-like features:1.3), dynamic flowing black hair, casual modern clothing, detailed beaded necklace, earbuds, (natural eye design:1.2), genuine cheerful expression, (semi-realistic shading:1.3), subtle skin details, atmospheric purple background",
            "pixar human character, young asian person, (realistic facial structure:1.3), detailed wavy black hair, contemporary casual wear, colorful beaded accessories, (lifelike animated eyes:1.2), authentic smile, (balanced stylization:1.3), natural lighting, purple mood background"
        ]

        # 픽사 스타일 프롬프트
        self.pixar_prompts = [
            "pixar 3D rendering style, young asian male, detailed skin texture, physically based rendering, subsurface scattering, volumetric lighting, modern casual wear with headphones and beaded necklace, high quality 3D model, expressive pixar-style eyes, purple ambient lighting",
            "pixar character design, asian youth, detailed hair simulation, modern outfit with accessories, signature pixar facial expressions, high-end 3D rendering, ambient occlusion, realistic materials, purple atmospheric background, studio lighting setup",
            "pixar animation style, detailed 3D model of asian male, high-fidelity hair rendering, casual modern clothing, pixar-style eye design, professional 3D character art, physically accurate materials, purple environment lighting, cinematic composition"
        ]
        
        # 지브리 스타일 프롬프트
        self.ghibli_prompts = [
            "studio ghibli style, young asian male with messy brown hair, wearing black shirt and colorful beaded necklace, round expressive eyes, warm genuine smile, soft natural lighting, gentle shadows, clean simple linework, muted color palette, purple background, traditional ghibli animation, slice of life aesthetic",
            "ghibli character design, asian youth with fluffy natural hair, casual black t-shirt, turquoise and orange beaded necklace, characteristic ghibli eyes, cheerful open smile, soft diffused lighting, simple clean animation style, subtle color gradients, purple atmospheric background, hayao miyazaki character art",
            "miyazaki animation style, asian male portrait, natural windswept hair, simple casual wear, beaded necklace detail, classic ghibli facial features, gentle eye design, warm friendly expression, smooth clean lines, subtle shading, purple toned background, authentic ghibli aesthetic"
        ]
        
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
    def normalize_path(self, path):
        """경로 정규화"""
        try:
            # 따옴표 제거 및 공백 제거
            path = path.strip().strip('"\'')
            # 백슬래시를 슬래시로 변환
            path = path.replace('\\', '/')
            # 경로 정규화
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
            
    def preprocess_image(self, image_path):
        """이미지 전처리 개선"""
        try:
            # 이미지 읽기
            image_array = np.fromfile(image_path, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
                
            # 이미지 크기 조정
            image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LANCZOS4)
            
            # 향상된 전처리
            # 1. 노이즈 제거 최적화
            image = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
            
            # 2. 선명도 향상 (자연스러운 디테일)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.4
            image = cv2.filter2D(image, -1, kernel)
            
            # 3. 자연스러운 대비 향상
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 4. 피부톤 개선
            image = cv2.convertScaleAbs(image, alpha=1.1, beta=5)
            
            # 5. 자연스러운 색상 보정
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.15)  # 적당한 채도 증가
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # BGR to RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(image)
            
            return pil_image
        except Exception as e:
            raise ValueError(f"이미지 전처리 중 오류 발생: {str(e)}")
            
    def post_process_image(self, image, enhance_strength=0.2):
        """이미지 후처리"""
        try:
            # PIL Image를 numpy 배열로 변환
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 부드러운 후처리
            # 1. 자연스러운 밝기와 대비
            image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
            
            # 2. 약한 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * enhance_strength
            image = cv2.filter2D(image, -1, kernel)
            
            # 3. 부드러운 블러 적용
            image = cv2.GaussianBlur(image, (3,3), 0.5)
            
            # 4. 자연스러운 채도 조정
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # 채도 약간 증가
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # 명도 약간 증가
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # PIL Image로 변환
            return Image.fromarray(image)
        except Exception as e:
            print(f"후처리 중 오류 발생: {str(e)}")
            return image

    def process_image(self, image_path, style):
        """이미지 처리"""
        try:
            # 스타일 선택에 따른 처리
            if style == '1':
                result = self.apply_disney_style(image_path)
            elif style == '2':
                result = self.apply_pixar_style(image_path)
            elif style == '3':
                result = self.apply_ghibli_style(image_path)
            else:
                raise ValueError("지원하지 않는 스타일입니다.")
            
            # 후처리 적용
            result = self.post_process_image(result, enhance_strength=0.2)
            return result
        except Exception as e:
            raise ValueError(f"이미지 처리 중 오류 발생: {str(e)}")

    def apply_disney_style(self, image_path):
        """디즈니/픽사 스타일 적용"""
        try:
            # 이미지 전처리
            pil_image = self.preprocess_image(image_path)
            
            # 랜덤 프롬프트 선택
            prompt = random.choice(self.disney_prompts)
            
            # 스타일 변환 (파라미터 최적화)
            result = self.disney_pipe(
                prompt=prompt,
                image=pil_image,
                strength=0.78,        # 사람다운 특징을 위해 약간 증가
                guidance_scale=8.0,   # 선명한 디테일 유지
                num_inference_steps=40,  # 충분한 품질의 렌더링
                negative_prompt="unrealistic anatomy, exaggerated features, cartoon, caricature, chibi, distorted proportions, flat shading, cell shading, low quality, blurry, deformed features, oversaturated colors, simple shading"
            ).images[0]
            
            return result
        except Exception as e:
            raise ValueError(f"디즈니/픽사 스타일 적용 중 오류 발생: {str(e)}")

    def apply_pixar_style(self, image_path):
        """픽사 스타일 적용"""
        try:
            # 이미지 전처리
            pil_image = self.preprocess_image(image_path)
            
            # 랜덤 프롬프트 선택
            prompt = random.choice(self.pixar_prompts)
            
            # 스타일 변환 (파라미터 최적화)
            result = self.disney_pipe(
                prompt=prompt,
                image=pil_image,
                strength=0.7,         # 3D 효과를 위해 약간 더 강한 변환
                guidance_scale=7.5,   # 균형잡힌 프롬프트 영향력
                num_inference_steps=35,  # 더 세밀한 렌더링을 위해 스텝 증가
                negative_prompt="2D, flat, anime, cartoon, sketch, drawing, low quality, blurry, deformed, distorted, dull, ugly, oversaturated"
            ).images[0]
            
            return result
        except Exception as e:
            raise ValueError(f"픽사 스타일 적용 중 오류 발생: {str(e)}")

    def apply_ghibli_style(self, image_path):
        """지브리 스타일 적용"""
        try:
            # 이미지 전처리
            pil_image = self.preprocess_image(image_path)
            
            # 랜덤 프롬프트 선택
            prompt = random.choice(self.ghibli_prompts)
            
            # 스타일 변환 (파라미터 최적화)
            result = self.ghibli_pipe(
                prompt=prompt,
                image=pil_image,
                strength=0.75,        # 지브리 스타일 특징을 더 강하게
                guidance_scale=6.5,   # 부드러운 변환을 위해 낮춤
                num_inference_steps=35,  # 디테일을 위해 스텝 증가
                negative_prompt="3D, CGI, photorealistic, disney style, pixar style, western animation, harsh shadows, oversaturated colors, high contrast, sharp edges, detailed textures, complex shading, dark shadows"
            ).images[0]
            
            return result
        except Exception as e:
            raise ValueError(f"지브리 스타일 적용 중 오류 발생: {str(e)}")

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
                filename = filename + '.png'  # 기본 확장자 추가
                ext = '.png'
            
            # 전체 경로 생성
            full_path = os.path.join(save_dir, filename)
            
            # PIL Image를 numpy 배열로 변환
            if isinstance(image, Image.Image):
                image = np.array(image)
                # RGB to BGR 변환
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 이미지 품질 설정
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]  # 최대 품질
            
            # 한글 경로 처리를 위해 cv2.imencode 사용
            success, encoded_image = cv2.imencode(ext, image, encode_param)
            if not success:
                raise ValueError("이미지 인코딩에 실패했습니다.")
            
            # 인코딩된 이미지 저장
            with open(full_path, 'wb') as f:
                f.write(encoded_image.tobytes())
            
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

    def detect_faces(self, image):
        """
        OpenCV Cascade Classifier 기반 얼굴 감지
        
        Args:
            image: BGR(OpenCV) 이미지
        Returns:
            faces: 얼굴 bounding box 리스트 [(x, y, w, h), ...]
        """
        faces = []
        
        if self.face_detector is not None:
            try:
                # 이미지 전처리
                height, width = image.shape[:2]
                
                # 그레이스케일 변환
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 히스토그램 균등화
                gray = cv2.equalizeHist(gray)
                
                # CLAHE 적용
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
                
                # 얼굴 감지
                faces_detected = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # 더 작은 값으로 조정
                    minNeighbors=6,    # 더 큰 값으로 조정
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces_detected) > 0:
                    for (x, y, w, h) in faces_detected:
                        faces.append((x, y, w, h))
                
            except Exception as e:
                print(f"얼굴 감지 중 오류 발생: {str(e)}")
        
        return faces

    def analyze_image(self, image_path):
        """이미지 분석을 통한 스타일 추천"""
        try:
            # 한글 경로 처리를 위해 np.fromfile과 cv2.imdecode 사용
            image_array = np.fromfile(image_path, np.uint8)
            # 직접 그레이스케일로 읽기
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")
            
            # 그레이스케일 변환 (BGR에서 직접 변환)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 이미지 전처리 개선
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # CLAHE 적용 (더 강한 대비)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 얼굴 감지
            faces = self.detect_faces(image)
            
            # RGB 변환 (스타일 분석용)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))
            
            # 주요 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:,:,1])
            value = np.mean(hsv[:,:,2])
            
            # 배경 분석
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (512 * 512)
            
            # 스타일 추천 로직
            if len(faces) > 0:
                # 얼굴이 있는 경우
                face_size = max([w * h for (x, y, w, h) in faces]) / (512 * 512)
                
                if saturation > 100 and value > 150:
                    # 밝고 채도가 높은 경우
                    if face_size > 0.2 and edge_density > 0.1:
                        print("선택된 스타일: 디즈니 스타일")
                        return '1'
                    else:
                        print("선택된 스타일: 지브리 스타일")
                        return '3'
                else:
                    # 자연스러운 톤의 경우
                    if face_size > 0.15 and edge_density > 0.15:
                        print("선택된 스타일: 디즈니 스타일")
                        return '1'
                    else:
                        print("선택된 스타일: 지브리 스타일")
                        return '3'
            else:
                # 얼굴이 없는 경우
                if saturation > 100 and edge_density > 0.2:
                    print("선택된 스타일: 디즈니 스타일")
                    return '1'
                else:
                    print("선택된 스타일: 지브리 스타일")
                    return '3'
                    
        except Exception as e:
            print(f"이미지 분석 중 오류 발생: {str(e)}")
            print("기본 스타일(디즈니 스타일)을 적용합니다.")
            return '1'

    def main(self):
        """메인 함수"""
        print("\n=== 애니메이션 스타일 변환기 ===")
        
        while True:
            try:
                # 이미지 경로 입력
                while True:
                    image_path = input("\n변환할 이미지의 경로를 입력하세요: ").strip()
                    if not image_path:
                        print("경로를 입력해주세요.")
                        continue
                    try:
                        # 경로 정규화 및 검증
                        success, abs_path = self.validate_image_path(image_path)
                        if not success:
                            print(abs_path)  # 오류 메시지 출력
                            continue
                        break
                    except Exception as e:
                        print(f"경로 처리 중 오류 발생: {str(e)}")
                        continue
                
                # 스타일 선택 방식 선택
                while True:
                    print("\n스타일 선택 방식을 선택하세요:")
                    print("1: 수동으로 스타일 선택")
                    print("2: 자동으로 스타일 분석")
                    choice = input("선택 (1 또는 2): ").strip()
                    
                    if choice in ['1', '2']:
                        break
                    print("1 또는 2를 입력해주세요.")
                
                # 스타일 선택
                if choice == '1':
                    # 수동 스타일 선택
                    print("\n사용 가능한 스타일:")
                    for key, value in self.styles.items():
                        print(f"{key}: {value}")
                    
                    while True:
                        style = input("\n스타일 번호를 선택하세요 (1-4): ").strip()
                        if style in self.styles:
                            print(f"\n선택된 스타일: {self.styles[style]}")
                            break
                        print("1, 2, 3, 또는 4를 입력해주세요.")
                else:
                    # 자동 스타일 분석
                    print("\n이미지 분석 중...")
                    style = self.analyze_image(abs_path)
                    print(f"\n분석 결과 추천 스타일: {self.styles[style]}")
                
                # 저장 경로 입력
                while True:
                    filename = input("\n저장할 파일명을 입력하세요 (확장자 없이): ").strip()
                    if not filename:
                        print("파일명을 입력해주세요.")
                        continue
                    try:
                        filename = filename + '.png'  # 기본 확장자 추가
                        break
                    except Exception as e:
                        print(f"경로 처리 중 오류 발생: {str(e)}")
                        continue
                
                # 이미지 처리
                try:
                    print("\n이미지 변환 중...")
                    result = self.process_image(abs_path, style)
                    self.save_image(result, filename)
                except Exception as e:
                    print(f"\n오류가 발생했습니다: {str(e)}")
                    print("다시 시도해주세요.")
                    continue
                
                # 계속할지 여부 확인
                while True:
                    choice = input("\n계속 변환하시겠습니까? (y/n): ").strip().lower()
                    if choice in ['y', 'n']:
                        break
                    print("y 또는 n을 입력해주세요.")
                
                if choice == 'n':
                    print("프로그램을 종료합니다.")
                    break
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n오류가 발생했습니다: {str(e)}")
                print("다시 시도해주세요.")
                continue

if __name__ == "__main__":
    try:
        stylizer = AnimationStylizer()
        stylizer.main()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n예기치 않은 오류가 발생했습니다: {str(e)}")
        sys.exit(1) 
