# @author trendelkamp
# modified by rudzinski
# JFR - 27 March 2015 - taken from stallone and converted to python

import numpy as np

#class ScaledElementSampler:

def modePoint( a, b, c, d ):

    s = 2.0*(a+b+c)
    x0 = (a+b)*d+(a+c)
    D2 = (x0**2)-4.0*a*d*(a+b+c)

    return 1.0 / s*(x0-np.sqrt(D2))


def densityScaled( x, a, b, c, d, xm ):

    return ((x/xm)**a) * (((1.0-x)/(1.0-xm))**b) * (((d-x)/(d-xm))**c)


def logdensityScaled( x, a, b, c, d, xm ):

    # Case a>0, b>0
    if ( (a>1e-15) and (b>1e-15) ):
        return  a*(np.log(x)-np.log(xm)) + b*(np.log(1.0-x)-np.log(1.0-xm)) + c*(np.log(d-x)-np.log(d-xm))
    # Case a=0, b>0
    elif ( (a<1e-15) and (b>1e-15) ):
        return  b*(np.log(1.0-x)-np.log(1.0-xm)) + c*(np.log(d-x)-np.log(d-xm))
    # Case a>0, b=0
    elif ( (a>1e-15) and (b<1e-15) ):
        return  a*(np.log(x)-np.log(xm)) + c*(np.log(d-x)-np.log(d-xm))
    # Case a=0, b=0
    else:
        return  c*(np.log(d-x)-np.log(d-xm))

def logdensity( x, a, b, c, d ):

    # Case a>0, b>0
    if ( (a>1e-15) and (b>1e-15) ):
        return  a*np.log(x) + b*np.log(1.0-x) + c*np.log(d-x)
    # Case a=0, b>0
    elif ( (a<1e-15) and (b>1e-15) ):
        return b*np.log(1.0-x) + c*np.log(d-x)
    # Case a>0, b=0
    elif(a>1e-15 and b<1e-15):
        return a*np.log(x) + c*np.log(d-x)
    # Case a=0, b=0
    else:
        return c*np.log(d-x)


def logdensityD1( x, a, b, c, d):

    # Case a>0, b>0
    if ( (a>1e-15) and (b>1e-15) ):
        return  a/x - b/(1.0-x) - c/(d-x)
    # Case a=0, b>0
    elif ( (a<1e-15) and (b>1e-15) ):
        return -b/(1.0-x) - c/(d-x)
    # Case a>0, b=0
    elif ( (a>1e-15) and (b<1e-15) ):
        return a/x - c/(d-x)
    # Case a=0, b=0
    else:
        return -c/(d-x)


def logdensityD2( x, a, b, c, d ):

    # Case a>0, b>0
    if ( (a>1e-15) and (b>1e-15) ):
        return  -a/(x**2) - b/((1.0-x)**2) - c/((d-x)**2)
    # Case a=0, b>0
    elif( (a<1e-15) and (b>1e-15) ):
        return -b/((1.0-x)**2) - c/((d-x)**2)
    # Case a>0, b=0
    elif ( (a>1e-15) and (b<1e-15) ):
        return -a/(x**2) - c/((d-x)**2)
    # Case a=0, b=0
    else:
        return -c/((d-x)**2)


def sampleExponentialRestricted( randE, upperBound ):

    accept = False
    E = 0.0
    while not accept:
        E = randE.exponential(1,1)
        accept = ( E <= upperBound )

    return E


def sampleThreePieces( randU, randE, p1, p2, p3, xl, ql, al, xu, qu, au, a, b, c, d, xm ):

    V = 0.0
    U = 0.0
    E = 0.0
    X = 0.0

    accept = False
    while not accept:
        V = randU.uniform(0,1,1)
        if ( V < p1 ):
            U = randU.uniform(0,1,1)
            E = sampleExponentialRestricted(randE, xl*al)
            X = -E/al+xl
            accept = ( (np.log(U) + ql + al*(X-xl)) <= logdensityScaled(X,a,b,c,d,xm) )
        elif ( V < p1+p2 ):
            U = randU.uniform(0,1,1)
            X = randU.uniform(xl,xu,1)
            accept = ( np.log(U) <= logdensityScaled(X,a,b,c,d,xm) )
        else:
            U = randU.uniform(0,1,1)
            E = sampleExponentialRestricted( randE, (1.0-xu)*(-au) )
            X = -E/au+xu
            accept = ( (np.log(U) + qu + au*(X-xu)) <= logdensityScaled(X,a,b,c,d,xm) )

    return X


def sample( randU, randE, a, b, c, d):

    X = 0.0
    U = 0.0

    xm = 0.0
    sigma = 0.0

    xl = 0.0
    ql = 0.0
    al = 0.0

    xu = 0.0
    qu = 0.0
    au = 0.0

    wl = 0.0
    wm = 0.0
    wu = 0.0
    w = 0.0

    pl = 0.0
    pm = 0.0
    pu = 0.0

    # Case c = 0
    if ( np.abs(c) < 1e-15 ):
        X = np.random.beta(a+1.0, b+1.0, 1) # JFR - should double check this, bypasses the RandomState
    # Case c > 0
    else:
        # Case d = 0
        if( np.abs(d-1.0)<1e-15 ):
            X = np.random.beta(a+1.0,b+c+1.0,1)
        # Case d > 0
        else:
            # Test for feasible rejection from the Beta-density
            if( c*np.log((d-1.0)/d) > np.log(0.8) ):
		U = randU.uniform(0,1,1)
                X = np.random.beta(a+1.0,b+1.0)
                while ( np.log(U) > c*np.log((d-X)/d) ):
                    U = randU.uniform(0,1,1)
                    X = np.random.beta(a+1.0,b+1.0)
            #Else use piecewise exponential bounding density
            else:
                # Case a>0, b>0
                if ( (np.abs(a)>1e-15) and (np.abs(b)>1e-15) ):
                    xm = modePoint(a,b,c,d)
                    sigma = 1.0 / np.sqrt(-logdensityD2(xm,a,b,c,d))
                    xl = max(0.0,xm-sigma)
                    xu = min(1.0,xm+sigma)
                    
                # Case a>0, b=0
                elif ( (np.abs(a)>1e-15) and (np.abs(b)<1e-15) ):
                    xm = min((a*d)/(a+c),1.0)
                    # Case xm < 1
                    if ( xm < 1 ):
                        sigma=1.0/np.sqrt(-logdensityD2(xm,a,b,c,d))
                        xl = max(0.0,xm-sigma)
                        xu = min(1.0,xm+sigma)
                    # Case xm >= 1
                    else:
                        xl = 1.0
                        xu = 1.0
                # Case a=0, b>0
                elif( (np.abs(a)<1e-15) and (np.abs(b)>1e-15) ):
                    xl = 0.0
                    xu = 0.0
                    xm = 0.0
                # Case a=0, b=0
                else:
                    xl = 0.0
                    xu = 0.0
                    xm = 0.0
                #Weight of the central enveloping piece (uniform)

                wm = xu - xl

                # Case xl > 0
                if ( xl > 0.0):
                    al = logdensityD1(xl,a,b,c,d)
                    ql = logdensityScaled(xl,a,b,c,d,xm)
                    wl = np.exp(ql) / al*(-1.0)*np.expm1(-al*xl)
                # Case xl = 0
                else:
                    xl = 0.0
                    wl = 0.0
                    al = float("inf")
                    ql = float("-inf") # JFR - Need to double check the application of these

                # Case xu < 1
                if ( xu < 1.0 ):
                    au = logdensityD1(xu,a,b,c,d)
                    qu = logdensityScaled(xu,a,b,c,d,xm)
                    wu = np.exp(qu) / au*np.expm1(au*(1.0-xu))
                # Case xu=1
                else:
                    xu = 1.0
                    wu = 0.0
                    au = float("-inf")
                    qu = float("-inf")

                # Compute normalized weights = probabilities
                w = wl + wm + wu
                pl = wl/w
                pm = wm/w
                pu = wu/w

                # Sample X
                X = sampleThreePieces(randU, randE, pl, pm, pu, xl, ql, al, xu, qu, au, a, b, c, d, xm)

    return X
