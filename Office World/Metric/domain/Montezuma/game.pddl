(define
    (domain Montezuma)
    (:requirements :typing :fluents)
    
    (:types
        loc key - object
        spot leftdoor rightdoor keyloc leftladder rightladder skull - loc
    )
    (:predicates
        (keyexist)
        (actorwithkey)
        (at ?loc - loc)
        (gameover)
    )
    (:functions (quality) (lastquality)
                )

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
                (not (at ?from) )
                 )
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
                (increase (quality) 1)

    ))
    (:action keyloc-to-leftladder
        :parameters (?from - keyloc ?to - leftladder)
        :precondition (and (at ?from) 
                        )
        :effect (and (at ?to)
                (not (at ?from))
        )
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
            (increase (quality) 3)
        )
    )  
    (:action getout
        :parameters ()
        :precondition (and 
        (> (quality) (lastquality))
        )        
        :effect (and 
        (gameover))
        )
    
)