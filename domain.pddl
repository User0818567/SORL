(define
    (domain Montezuma)
    (:requirements :typing :fluents)
    
    (:types
        loc - object
        spot leftdoor rightdoor keyloc leftladder rightladder skull - loc
    )
    (:predicates
        (keyexist)
        (actorwithkey)
        (at ?loc - loc)
        (gameover)
    )
    (:functions (quality))

    (:action spot-to-rightladder
        :parameters (?from - spot ?to - rightladder)
        :precondition (and (at ?from) 
                        )
        :effect (and (at ?to)
                (not (at ?from))
    ))
    (:action rightladder-to-leftladder
        :parameters (?from - rightladder ?to - leftladder)
        :precondition (and (at ?from) )
        :effect (and (at ?to)
                (not (at ?from)) )
    )
    (:action leftladder-to-keyloc
        :parameters (?from - leftladder ?to - keyloc)
        :precondition (and (at ?from) 
                            (not (actorwithkey))
                            (keyexist)
                        )
        :effect (and (at ?to)
                (not (at ?from))
                (not (keyexist))
                (actorwithkey)
                (decrease (quality) 100)
    ))
    (:action keyloc-to-leftladder
        :parameters (?from - keyloc ?to - leftladder)
        :precondition (and (at ?from) 
                        )
        :effect (and (at ?to)
                (not (at ?from)))
    )
    (:action leftladder-to-rightladder
        :parameters (?from - leftladder ?to - rightladder)
        :precondition (and (at ?from) 
        )
        :effect (and 
            (not (at ?from))
            (at ?to)
        )
    )
    (:action rightladder-to-spot
        :parameters (?from - rightladder ?to - spot)
        :precondition (and (at ?from) 
         )
        :effect (and 
            (not (at ?from))
            (at ?to)
        )
    )    
    (:action spot-to-rightdoor
        :parameters (?from - spot ?to - rightdoor)
        :precondition (and (at ?from) 
        )
        :effect (and 
            (not (at ?from))
            (at ?to)
            (decrease (quality) 300)
        )
    )  
    (:action getout
        :parameters (?rightdoor)
        :precondition (and 
        (< (quality) -200)
        (at ?rightdoor))
        :effect (and 
        (gameover))
    )
    
)