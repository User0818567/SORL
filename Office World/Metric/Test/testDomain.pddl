(define
    (domain OfficeWorld)
    (:requirements :typing :fluents)
    (:predicates
        (haveCoffee)
        (haveMail)
        (deliveredCoffee)
        (deliveredMail)
    )
    (:functions (quality))
    	(:action act0
		:precondition (and (not (haveCoffee)) (not (haveMail)) (not (deliveredCoffee)) (not (deliveredMail)) )
		:effect (and (haveMail) (increase (quality) 0 ) )
	)
	(:action act1
		:precondition (and (not (haveCoffee)) (haveMail) (not (deliveredCoffee)) (not (deliveredMail)) )
		:effect (and (not (haveMail)) (deliveredMail) (increase (quality) 100 ) )
	)
)