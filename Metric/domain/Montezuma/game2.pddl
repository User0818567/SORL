(define
    (domain Montezuma)
    (:requirements :typing :fluents)
    
    (:types
       
    )
    (:predicates
        (keyexist)
        (actorwithkey)
        
        (atspot)
        (atleftdoor)
        (atrightdoor)
        (atkeyloc)
        (atleftladder)
        (atrightladder)
        (gameover)
    )
    (:functions (quality))

    (:action spot-to-rightladder
        :parameters ()
        :precondition (and (atspot) 
                        )
        :effect (and (atrightladder)
                (not (atspot))
    ))
    (:action rightladder-to-leftladder
        :parameters ()
        :precondition (and (atrightladder) )
        :effect (and (atleftladder)
                (not (atrightladder)) )
    )
    (:action leftladder-to-keyloc
        :parameters ()
        :precondition (and (atleftladder) 
                            (not (actorwithkey))
                            (keyexist)
                        )
        :effect (and (atkeyloc)
                (not (atleftladder))
                (not (keyexist))
                (actorwithkey)
                (decrease (quality) 100)
    ))
    (:action keyloc-to-leftladder
        :parameters ()
        :precondition (and (atkeyloc) 
                        )
        :effect (and (atleftladder)
                (not (atkeyloc)))
    )
    (:action leftladder-to-rightladder
        :parameters ()
        :precondition (and (atleftladder) 
        )
        :effect (and 
            (not (atleftladder))
            (atrightladder)
        )
    )
    (:action rightladder-to-spot
        :parameters ()
        :precondition (and (atrightladder) 
         )
        :effect (and 
            (not (atrightladder))
            (atspot)
        )
    )    
    (:action spot-to-rightdoor
        :parameters ()
        :precondition (and (atspot) 
        )
        :effect (and 
            (not (atspot))
            (atrightdoor)
            
        )
    )  
    (:action getout
        :parameters ()
        :precondition (and 
        (atrightdoor))
        :effect (and 
        (gameover)
        (decrease (quality) 300))
    )
)