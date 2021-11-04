(define
    (domain Montezumatest)
    (:requirements :typing :fluents)
    (:predicates
        (keyexist)
        (atleftdoor)
        (atrightdoor)
        (atkey)
        (atmiddleladder)
        (atleftladder)
        (atrightladder)
    )
    (:functions (quality))
    (:action middleladder-to-rightladder
        :precondition (and (atmiddleladder))
        :effect (and (atrightladder)
                (not (atmiddleladder)))
    )

    (:action rightladder-to-leftladder
        :precondition (and (atrightladder))
        :effect (and (atleftladder)
                (not (atrightladder) ) )
    )

    (:action leftladder-to-key
        :precondition (and (atleftladder))
        :effect (and (atkey)
                (not (atleftladder) ) )
    )

    (:action get-key
        :precondition (and (atkey)
                            (keyexist))
        :effect (and (not (keyexist))
                (increase (quality) 100))
    )

    (:action key-to-leftladder
        :precondition (and (atkey))
        :effect (and (atleftladder)
                (not (atkey)))
    )

    (:action leftladder-to-middleladder
        :precondition (and (atleftladder))
        :effect (and (atmiddleladder)
                (not (atleftladder)))
    )
    
    (:action middleladder-to-rightdoor
        :precondition (and (atmiddleladder))
        :effect (and (atrightdoor)
                (not (atmiddleladder)))
    )

    (:action getout
        :precondition (and (atrightdoor))
        :effect (and (not (atrightdoor))
                (increase (quality) 300))
    )
)