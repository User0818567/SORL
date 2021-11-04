(define (problem problem1) 
(:domain Montezumatest)
(:metric minimize ( quality) )
(:goal (and(atRightDoor)
(not (keyexist))

(> (quality) 401.0)
))
(:init 	(keyexist)
	(atMiddleLadder)
	(= (quality) 0) )

)