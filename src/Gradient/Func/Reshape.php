<?php
namespace Rindow\NeuralNetworks\Gradient\Func;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Gradient\Core\AbstractFunction;

class Reshape extends AbstractFunction
{
    protected $numOfInputs = 2;

    protected function call(array $inputs) : array
    {
        $K = $this->backend;
        $inp = $inputs[0];
        $shape = $inputs[1];
        $shape = $K->ndarray($shape);
        $inpShape = $inp->shape();

        $container = $this->container();
        $container->inpShape = $inpShape;
        $container->shapeShape = $shape->shape();
        $container->shapeDtype = $shape->dtype();

        if($shape->shape()===[]) {
            if($inpShape!==[1] && $inpShape!==[]) {
                throw new InvalidArgumentException(
                    'Shape is an invalid size specification.'.
                    ' input-shape:['.implode(',',$inpShape).'],'.
                    ' target-shape:[]');
            }
            $out = $inp->reshape([]);
            return [$out];
        }
        $tmpShape = [];
        $fixedSize = 1;
        $countFlatten = 0;
        foreach($shape as $dim) {
            if($dim==0) {
                $dim = $inpShape[0];
            }
            if($dim!=-1) {
                $fixedSize *= $dim;
            } else {
                $countFlatten++;
            }
            $tmpShape[] = $dim;
        }
        $inpSize = (int)array_product($inpShape);
        if($inpSize%$fixedSize != 0 || $countFlatten>1) {
            $strTarShape = array_map(fn($x)=>($x==-1)?'?':$x, $tmpShape);
            throw new InvalidArgumentException(
                'Shape is an invalid size specification.'.
                ' input-shape:['.implode(',',$inpShape).'],'.
                ' target-shape:['.implode(',',$strTarShape).']');
        }
        $flatten = (int)($inpSize/$fixedSize);
        foreach($tmpShape as $dim) {
            if($dim==-1) {
                $targetShape[] = $flatten;
            } else {
                $targetShape[] = $dim;
            }
        }
        $out = $inp->reshape($targetShape);

        return [$out];
    }

    protected function differentiate(array $dOutputs) : array
    {
        $K = $this->backend;
        $container = $this->container();
        $inpShape = $container->inpShape;
        $shapeShape = $container->shapeShape;
        $shapeDtype = $container->shapeDtype;

        $dInputs = $dOutputs[0]->reshape($inpShape);
        $dShape = $K->zeros($shapeShape,$shapeDtype);
        return [$dInputs,$dShape];
    }
}
