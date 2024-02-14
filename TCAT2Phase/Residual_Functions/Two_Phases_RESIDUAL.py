from TCAT2Phase.Residual_Functions import _Two_Phases_RESIDUAL 
__all__ =[
  "test"
]

def test():
    ans = _Two_Phases_RESIDUAL.cython_test()
    print(ans)
    return True
