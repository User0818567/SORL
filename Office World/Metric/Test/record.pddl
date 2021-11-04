    
	
	
	
	
task3
	
	(:action act0
		:precondition (and (not (haveCoffee)) (not (haveMail)) (not (deliveredCoffee)) (not (deliveredMail)) )
		:effect (and (haveCoffee) (increase (quality) 0 ) )
	)
	(:action act1
		:precondition (and (haveCoffee) (not (haveMail)) (not (deliveredCoffee)) (not (deliveredMail)) )
		:effect (and (haveMail) (increase (quality) 2 ) )
	)
	(:action act2
		:precondition (and (haveCoffee) (haveMail) (not (deliveredCoffee)) (not (deliveredMail)) )
		:effect (and (not (haveCoffee)) (not (haveMail)) (deliveredCoffee) (deliveredMail) (increase (quality) 100 ) )
	)

task1

	(:action act0
		:precondition (and (not (haveCoffee)) (not (deliveredCoffee)) )
		:effect (and (haveCoffee) (increase (quality) 0 ) )
	)
	(:action act1
		:precondition (and (haveCoffee) (not (deliveredCoffee)) )
		:effect (and (not (haveCoffee)) (deliveredCoffee) (increase (quality) 100 ) )
	)


task2

(:action act0
:precondition (and (not (haveMail)) (not (deliveredMail)) )
:effect (and (haveMail) (increase (quality) 0 ) ))
(:action act1
:precondition (and (haveMail) (not (deliveredMail)) )
:effect (and (not (haveMail)) (deliveredMail) (increase (quality) 100 ) ))

	