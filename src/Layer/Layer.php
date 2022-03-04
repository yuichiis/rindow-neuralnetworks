<?php
namespace Rindow\NeuralNetworks\Layer;

use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;

/**
 *
 */
interface Layer extends LayerBase
{
    public function forward(object $inputs, bool $training);
    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array;
}
