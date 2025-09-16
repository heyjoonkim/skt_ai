def function_QT(enable: bool = None, get: bool = False):
    """
    음성 인식(Voice Recognition) 기능을 제어하거나 현재 상태를 조회하는 함수.

    설명:
    - enable 파라미터를 통해 음성 인식 기능을 켜거나 끌 수 있음
    - get=True 로 설정하면 현재 음성 인식 기능 상태를 조회함
    - enable와 get은 상호 배제 관계이며, 동시에 True/None 이 될 수 없음
    - enable가 None이고 get=False인 경우, 특별 동작 없이 함수 호출은 무의미함

    Parameters:
    - enable (bool, optional): 음성 인식 기능 on/off 제어 (get=True인 경우 None이어야 함)
        * True  → 음성 인식 기능 켜기
        * False → 음성 인식 기능 끄기
        * None  → 제어 동작 없음 (상태 조회인 경우 사용 안 함)
    - get (bool): 음성 인식 기능 상태 조회 여부 (enable이 None일 때만 True로 설정 가능)
        * True  → 현재 음성 인식 기능 상태 반환
        * False → 상태 조회하지 않음

    동의어:
    - 활성화: 켜줘, 시작해, 켜져있어, 상태 알려줘, 상태가 어때
    - 비활성화: 꺼줘, 중지해, 멈춰줘, 종료해

    우선순위 규칙:
    - get=True 우선 처리: 상태 조회가 명확한 경우 enable 파라미터는 None 이어야 함
    - enable 설정 시 get=False 로 적용

    반환 스키마:
    - get=True 일 때 : 현재 음성 인식 상태를 bool 값으로 반환
        * True → 기능 켜짐
        * False → 기능 꺼짐
    - enable 설정 시 : 제어 성공 여부 또는 상태 반영 결과 반환

    예외 처리:
    - enable과 get이 동시에 의미 있는 값으로 주어지면 get 우선 처리하며 enable 무시
    - enable=None, get=False인 호출은 무시하거나 오류 처리 권장
    """
    pass