(define (problem problem1) 
(:domain Montezumatest)

(:metric minimize( quality) )
(:goal (and
    (>= (quality) 400)
))
(:init	(keyexist)
	(atmiddleladder)
	(= (quality) 0) )

)
