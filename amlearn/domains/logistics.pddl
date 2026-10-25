; Logistics, untyped STRIPS, as in the 1998 International Planning Competition.
(define (domain logistics)
  (:requirements :strips)
  (:predicates (obj ?o) (truck ?t) (location ?l) (airplane ?a) (city ?c) (airport ?l)
               (at ?x ?l) (in ?o ?v) (in-city ?l ?c))
  (:action load-truck
    :parameters (?o ?t ?l)
    :precondition (and (obj ?o) (truck ?t) (location ?l) (at ?t ?l) (at ?o ?l))
    :effect (and (not (at ?o ?l)) (in ?o ?t)))
  (:action load-airplane
    :parameters (?o ?a ?l)
    :precondition (and (obj ?o) (airplane ?a) (location ?l) (at ?o ?l) (at ?a ?l))
    :effect (and (not (at ?o ?l)) (in ?o ?a)))
  (:action unload-truck
    :parameters (?o ?t ?l)
    :precondition (and (obj ?o) (truck ?t) (location ?l) (at ?t ?l) (in ?o ?t))
    :effect (and (not (in ?o ?t)) (at ?o ?l)))
  (:action unload-airplane
    :parameters (?o ?a ?l)
    :precondition (and (obj ?o) (airplane ?a) (location ?l) (in ?o ?a) (at ?a ?l))
    :effect (and (not (in ?o ?a)) (at ?o ?l)))
  (:action drive-truck
    :parameters (?t ?from ?to ?c)
    :precondition (and (truck ?t) (location ?from) (location ?to) (city ?c) (at ?t ?from)
                       (in-city ?from ?c) (in-city ?to ?c))
    :effect (and (not (at ?t ?from)) (at ?t ?to)))
  (:action fly-airplane
    :parameters (?a ?from ?to)
    :precondition (and (airplane ?a) (airport ?from) (airport ?to) (at ?a ?from))
    :effect (and (not (at ?a ?from)) (at ?a ?to))))
