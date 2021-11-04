(define (problem problem1) 
(:domain OfficeWorld)

(:metric minimize( quality) )
(:goal (and
    (> (quality) 0)
))
(:init	
	(= (quality) 0) )

)
