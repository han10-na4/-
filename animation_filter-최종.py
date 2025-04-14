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
