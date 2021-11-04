(define (problem problem1) 
(:domain OfficeWorld)
(:metric minimize ( quality) )
(:goal (and
(> (quality) 100.0)
))
(:init 	(not (haveCoffee))
	(not (haveMail))
	(not (deliveredCoffee))
	(not (deliveredMail))
	(= (quality) 0) )

)