<?php
namespace Rindow\NeuralNetworks\Model;
interface Model
{
    public function compile(array $options=null) : void;
    public function fit(NDArray $inputs, NDArray $tests, array $options=null) : array;
    public function evaluate(NDArray $x, NDArray $t, array $options=null) : array;
    public function predict($inputs, array $options=null) : NDArray;
    public function toJson() : string;
    public function saveWeights(&$modelWeights,$portable=null) : void;
    public function loadWeights($modelWeights) : void;
    public function save($filepath,$portable=null) : void;
}