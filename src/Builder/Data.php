<?php
namespace Rindow\NeuralNetworks\Builder;

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Data\Dataset\NDArrayDataset;
use Rindow\NeuralNetworks\Data\Dataset\CSVDataset;
use Rindow\NeuralNetworks\Data\Image\ImageFilter;
use LogicException;

class Data
{
    protected $matrixOperator;

    public function __construct($matrixOperator)
    {
        $this->matrixOperator = $matrixOperator;
    }

    public function __get( string $name )
    {
        if(!method_exists($this,$name)) {
            throw new LogicException('Unknown dataset: '.$name);
        }
        return $this->$name();
    }

    public function __set( string $name, $value ) : void
    {
        throw new LogicException('Invalid operation to set');
    }

    public function NDArray(NDArray $inputs, array $options=null)
    {
        return new NDArrayDataset($this->matrixOperator, $inputs, $options);
    }

    public function CSV(string $path, array $options=null)
    {
        return new CSVDataset($this->matrixOperator, $path, $options);
    }

    public function ImageFilter(array $options=null)
    {
        return new ImageFilter($this->matrixOperator, $options);
    }

    public function ImageGenerator(NDArray $inputs, array $options=null)
    {
        $leftargs = [];
        $filter = new ImageFilter($this->matrixOperator, $options, $leftargs);
        $leftargs['filter']=$filter;
        return new NDArrayDataset($this->matrixOperator, $inputs, $leftargs);
    }
}
