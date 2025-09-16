def function_BP(enable: int = None, type: int = None, get: bool = False):
    """
    에이아이 모드(AI Mode)를 제어하거나 상태를 조회하는 함수.

    설명:
    - enable 파라미터는 기능 활성화 및 비활성화를 제어
      * enable=0: 기능 활성화 (켜기, 시작하기, 작동시키기)
      * enable=1: 기능 비활성화 (끄기, 중단하기, 종료하기)
      * None: enable 조작 없음
    - type 파라미터는 제어할 AI 모드의 구체적 유형을 지정
      * -1: 기본 모드 (일반 AI 모드)
      * 0: 실외 대기질 연동 모드
      * 1: 공간별 누적 데이터 모드
      * 2: 전체/복합 모드 (둘 다)
      * None: 모든 모드에 대해 동작하지 않음 (enable 있을 경우 기본 모드 대상)
    - get 파라미터는 현재 AI 모드 상태 조회 여부 지정
      * True: AI 모드 활성 상태를 조회 (enable 및 type 무시)
      * False: 상태 조회하지 않음

    추가 규칙:
    - get=True이면 enable, type 값은 무시됨
    - enable, type 둘 다 None 이면 아무 동작도 하지 않음
    - enable과 함께 type이 없으면 type은 기본값 -1로 설정됨 (기본 모드 대상)
    - type 값 중 2 (전체/복합 모드)은 "둘 다", "모두" 등 복수 표현과 매칭됨
    - 동의어 정리
      * 켜다 (켜줘, 온, 시작, 작동 등) → enable=0
      * 끄다 (꺼줘, 꺼, 오프, 중단, 종료, 끄고 제발 등) → enable=1
      * 실외 대기질 연동, 외부 대기 상태 연동, 외부 공기질 데이터 등은 type=0
      * 공간별 누적 데이터, 공간별 공기질 누적, 공기질 이력 연동 등은 type=1
      * 둘 다, 모두, 복합 모드 등은 type=2
      * 기본 모드(명시 없거나 단순 온/오프 요청)는 type=-1
    - 호출 시 get=True이면 enable, type은 None으로 설정해야 함 (상태 조회 모드)
    - enable, type 동시 지정 시 enable과 함께 해당 type 모드만 제어
    - enable 값 우선순위로, get 가 True 일 때만 조회로 작동

    Parameters:
    - enable (int or None): 기능 on/off 제어 값
        * 0 → 기능 활성화 (켜기, 시작 등)
        * 1 → 기능 비활성화 (끄기, 종료 등)
        * None → 조작 없음
    - type (int or None): 실행 모드 구분 값
        * -1 → 기본 모드 (명시 없을 때 기본값)
        * 0 → 실외 대기질 연동
        * 1 → 공간별 누적 데이터 연동
        * 2 → 전체/복합 모드 (둘 다)
        * None → 대상 모드 지정 없음 (enable 있을 때 기본 모드로 간주)
    - get (bool): 현재 모드 상태 조회 여부
        * True → 모드 활성 상태 조회 (enable, type 무시)
        * False → 조회하지 않음 (enable, type에 따라 제어)

    반환값:
    - 없음 (void). 함수 호출에 따른 AI 모드 제어 또는 상태 조회 트리거.

    예:
    - function_BP(enable=0, type=0)  # 실외 대기질 연동 모드 활성화
    - function_BP(enable=1, type=2)  # 전체 모드 비활성화
    - function_BP(get=True)           # AI 모드 활성 상태 조회
    """
    pass