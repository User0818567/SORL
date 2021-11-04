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
    (:action get-coffee
        :precondition (and )
        :effect (and (haveCoffee)
                )
    )

    (:action get-office
        :precondition (and (haveCoffee))
        :effect (and (not (haveCoffee))
                (deliveredCoffee)
                (increase (quality) 100) 
                )
    )

)